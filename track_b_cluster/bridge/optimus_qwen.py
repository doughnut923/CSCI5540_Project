"""Train the team's Qwen2-VL with a colocated, whole-encoder PP schedule.

This is an Optimus-inspired PP-only implementation, not full ATC Optimus.
Reuses ManifestBatches, model_provider, mcore_argv and MainGradAdamW. The
serial-reference arm uses the same model and optimizer for correctness checks.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist

from optimus_runtime import (PlanExecutor, Transport, collect_main_grads,
                             global_grad_norm, sample_indices, sync_encoder_grads, zero_grads)
from optimus_schedule import choose_plan, make_plan
from optimus_config import llm_layer_counts, model_arguments, whole_encoder_config
from optimus_experiment import (OptimusBatches, check_plan_config, check_step_inputs,
                                file_sha256)
from optimus_optimizer import MainGradAdamW
from optimus_profiling import StepProfiler


class QwenAdapter:
    dtype = torch.bfloat16

    def __init__(self, model, data):
        self.model, self.data = model, data

    def encoder_shape(self, m):
        return (self.data[m]["input_observation"]["visual_tokens_total"], self.model.config.hidden_size)

    def encoder_gradient_ranges(self, m):
        counts = self.data[m]["input_observation"]["visual_tokens_per_image"]
        offset, ranges = 0, []
        for count in counts:
            ranges.append((offset, offset + count))
            offset += count
        return ranges

    def accumulate_parameter_grads(self, kind):
        # Ordinary BF16 .grad buffers must not accumulate multiple microbatches
        # before conversion: that makes rounding depend on replica placement.
        module = self.model.vision_model if kind == "EB" else self.model.language_model
        collect_main_grads(module)

    def hidden_shape(self, m):
        return (self.data[m]["tokens"].shape[1], 1, self.model.config.hidden_size)

    def encoder_forward(self, m):
        b = self.data[m]
        return self.model.vision_model(vision_data=b["pixels"], grid_thw=b["grid"])

    def llm_forward(self, m, hidden, embedding):
        b = self.data[m]
        language = self.model.language_model
        if dist.get_rank() == 0:
            hidden = language.embedding(input_ids=b["tokens"], position_ids=None,
                                        image_input_mask=b["image_mask"].T,
                                        video_input_mask=None, image_embeds=embedding,
                                        video_embeds=None)
        else:
            language.set_input_tensor(hidden)
        return language(input_ids=None, position_ids=b["position_ids"],
                        attention_mask=None, decoder_input=hidden if dist.get_rank() == 0 else None,
                        labels=b["target"])

    def loss(self, m, output):
        mask = self.data[m]["loss_mask"].reshape(-1).float()
        if self.data[m]["input_observation"]["supervised_tokens"] < 1:
            raise ValueError("microbatch has no supervised tokens")
        return (output.float().reshape(-1) * mask).sum() / mask.sum()


def build_uneven_qwen(qwen, args, pre_process, post_process):
    # The pinned provider copies the decoder split into its vision config.
    # Construct with the same PMP specs and isolate the full vision tower here.
    qwen.build_tokenizer(args)
    args.rotary_seq_len_interpolation_factor = None
    language_config = qwen.core_transformer_config_from_args(args, qwen.Qwen2VLTransformerConfig)
    vision_config = whole_encoder_config(qwen, args, language_config)
    projector_config = qwen.get_vision_projection_config(
        deepcopy(language_config), vision_config.hidden_size, args.spatial_merge_size)
    model = qwen.Qwen2VLModel(
        language_transformer_config=language_config,
        language_transformer_layer_spec=qwen.get_gpt_layer_with_transformer_engine_spec(args.qk_layernorm),
        language_vocab_size=args.padded_vocab_size,
        language_max_sequence_length=args.max_position_embeddings,
        vision_transformer_config=vision_config,
        vision_transformer_layer_spec=qwen.get_qwen2vl_vision_model_spec(),
        drop_vision_class_token=False,
        vision_projection_config=projector_config,
        vision_projection_layer_spec=qwen.get_mlp_module_spec(add_norm=False).submodules,
        vision_projection_type="mlp",
        language_position_embedding_type=args.position_embedding_type,
        language_rotary_percent=args.rotary_percent,
        language_rotary_base=args.rotary_base,
        pre_process=pre_process, post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        language_share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
    )
    model.freeze(freeze_language_model=args.freeze_LM, freeze_vision_model=args.freeze_ViT,
                 freeze_vision_projection=False)
    return model


def build_model(qwen, args, rank, layer_counts, device):
    pre_process, post_process = rank == 0, rank == len(layer_counts) - 1
    if len(set(layer_counts)) == 1:
        model = qwen.model_provider(pre_process=pre_process, post_process=post_process)
    else:
        model = build_uneven_qwen(qwen, args, pre_process, post_process)
    if model.vision_model is None:
        # Same constructor/configuration as the pinned team model_provider; only
        # placement differs. No duplicate language model is built on this rank.
        from megatron_patch.model.qwen2_vl.visionmodel import Qwen2VisionModel
        vc = whole_encoder_config(qwen, args, model.config)
        pc = qwen.get_vision_projection_config(deepcopy(model.config), vc.hidden_size,
                                               args.spatial_merge_size)
        model.vision_model = Qwen2VisionModel(
            vc, qwen.get_qwen2vl_vision_model_spec(), pc,
            qwen.get_mlp_module_spec(add_norm=False).submodules,
            projection_type="mlp", pre_process=True, post_process=True)
    model.to(device).train()
    for tensor in list(model.vision_model.parameters()) + list(model.vision_model.buffers()):
        dist.broadcast(tensor.data, src=0)
    from megatron.core.distributed import DistributedDataParallelConfig
    model.ddp_config = DistributedDataParallelConfig()
    model.finish_grad_sync = lambda: None
    for p in model.parameters():
        if p.requires_grad:
            p.main_grad = torch.zeros_like(p, dtype=torch.float32)
    return model


def sample_signature(data):
    # Actual processed shapes; never group merely by a claimed token count.
    return tuple((tuple(b["pixels"].shape), tuple(b["grid"].flatten().tolist()),
                  tuple(b["tokens"].shape)) for b in data)


def sample_encoder(encoder, count, *, gradients=False):
    """Copy the same coordinates before and after a gradient reduction."""
    chunks = {}
    for name, param in encoder.named_parameters():
        if param.requires_grad:
            tensor = (param.main_grad if gradients else param).detach().flatten()
            indices = sample_indices(tensor.numel(), count, tensor.device)
            chunks[name] = tensor.index_select(0, indices).float()
    values = torch.cat(list(chunks.values())).cpu().tolist()
    snapshot, offset = {}, 0
    for name, chunk in chunks.items():
        end = offset + chunk.numel()
        snapshot[name] = values[offset:end]
        offset = end
    return snapshot


def profile_costs(result, world, owners):
    count = len(owners)
    costs = {kind: [[None] * count for _ in range(world)] for kind in ("EF", "EB", "F", "B")}
    for iv in result["compute_intervals"]:
        # CUDA events cover model calls only, excluding explicit P2P waits.
        # These are whole-encoder/stage costs, not a kernel-level profile.
        costs[iv["kind"]][iv["rank"]][iv["microbatch"]] = max(
            (iv["end_s"] - iv["start_s"]) * 1000, 1e-6)
    for kind in ("EF", "EB"):
        per_microbatch = [costs[kind][owner][m] for m, owner in enumerate(owners)]
        costs[kind] = [per_microbatch[:] for _ in range(world)]
    return costs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--warmup-steps", type=int, default=1)
    p.add_argument("--microbatches", type=int, default=4)
    p.add_argument("--layers", type=int, default=28)
    p.add_argument("--seq-length", type=int, default=1024,
                   help="Match the team B/C runner's maximum supported sequence length")
    p.add_argument("--input-plan", type=Path,
                   help="Require the audited sample plan prepared before GPU execution")
    p.add_argument("--memory-headroom-gb", type=float, default=2.0)
    p.add_argument("--optimizer-device", choices=("cuda", "cpu"), default="cuda",
                   help="CPU offloads FP32 master weights and AdamW states for the two-GPU quota")
    p.add_argument("--validation-samples", type=int, default=0,
                   help="Save evenly spaced parameter/used-gradient samples after timing")
    p.add_argument("--trace-first-encoder-gradients", action="store_true",
                   help="Diagnostic only: sample initial encoder weights and first-step gradients before and after reduction")
    p.add_argument("--schedule", choices=("serial-reference", "optimus-pp"), default="optimus-pp")
    p.add_argument("--batching", choices=("bucket", "manifest"), default="manifest",
                   help="Manifest order preserves mixed sizes/counts within each step")
    p.add_argument("--encoder-placement", choices=("profiled", "round-robin"), default="profiled",
                   help="Use round-robin in both arms to isolate scheduling from placement arithmetic")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--profile-step-phases", action="store_true",
                   help="Diagnostic synchronized phase timings; adds CUDA synchronization")
    opts = p.parse_args()
    if opts.steps < 1 or opts.warmup_steps < 0 or opts.microbatches < 1 or opts.validation_samples < 0:
        p.error("steps/microbatches must be positive; warmup must be nonnegative")
    if opts.trace_first_encoder_gradients and not opts.validation_samples:
        p.error("encoder gradient tracing requires --validation-samples")
    if not math.isfinite(opts.memory_headroom_gb) or opts.memory_headroom_gb < 0:
        p.error("memory headroom must be finite and nonnegative")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    try:
        layer_counts = llm_layer_counts(opts.layers, world)
    except ValueError as error:
        p.error(str(error))
    out = opts.output_dir.resolve()
    if int(os.environ.get("RANK", "0")) == 0:
        out.mkdir(parents=True, exist_ok=False)
    os.environ.update(QWEN2VL_HF_DIR=str(opts.model_dir), PP_SIZE=str(world),
                      NUM_LAYERS=str(opts.layers), NUM_MICROBATCHES=str(opts.microbatches))
    # The colocated executor owns scheduling; never install a second PP executor.
    os.environ.update(USE_PP_EXECUTOR="0", MEGATRON_USE_PP_EXECUTOR="0")
    if opts.deterministic:
        os.environ.update(NCCL_ALGO="Ring", NVTE_ALLOW_NONDETERMINISTIC_ALGO="0",
                          CUBLAS_WORKSPACE_CONFIG=":4096:8")
    sys.argv = ["optimus_qwen"] + model_arguments(opts.seq_length, layer_counts=layer_counts)
    if opts.deterministic:
        sys.argv.append("--deterministic-mode")
    from megatron.training.initialize import initialize_megatron
    from megatron_patch.arguments import get_patch_args
    initialize_megatron(extra_args_provider=get_patch_args)
    from megatron.training import get_args
    import qwen2_vl.pretrain_qwen as qwen
    from megatron.core import parallel_state
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    if (parallel_state.get_tensor_model_parallel_world_size() != 1
            or parallel_state.get_data_parallel_world_size() != 1):
        raise ValueError("this first implementation supports LLM TP=DP=1 only")
    rank = dist.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    args = get_args()
    model = build_model(qwen, args, rank, layer_counts, device)
    language_layers = [layer.layer_number for layer in model.language_model.decoder.layers]
    first_layer = 1 + sum(layer_counts[:rank])
    expected = list(range(first_layer, first_layer + layer_counts[rank]))
    if language_layers != expected:
        raise RuntimeError(f"rank {rank}: LLM layers {language_layers}, expected {expected}")
    vision_layers = [layer.layer_number for layer in model.vision_model.decoder.layers]
    if vision_layers != list(range(1, model.vision_model.config.num_layers + 1)):
        raise RuntimeError(f"rank {rank}: incomplete vision encoder layers {vision_layers}")
    transport = Transport(world, device)
    # AdamW needs FP32 masters and two moments in addition to the model and
    # main_grad already allocated. Check every rank before any starts training.
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    extra_bytes = parameter_count * 12 if opts.optimizer_device == "cuda" else 0
    free_bytes, _ = torch.cuda.mem_get_info()
    memory = {"rank": rank, "parameters": parameter_count,
              "llm_layer_numbers": language_layers, "encoder_layers": len(vision_layers),
              "free_bytes_before_optimizer": free_bytes,
              "optimizer_extra_bytes": extra_bytes,
              "fits_with_headroom": extra_bytes + opts.memory_headroom_gb * 2**30 < free_bytes}
    memory_by_rank = [None] * world
    dist.all_gather_object(memory_by_rank, memory)
    if not all(item["fits_with_headroom"] for item in memory_by_rank):
        raise RuntimeError("Colocated full-AdamW state needs more memory; use more PP GPUs "
                           "or a shared memory optimization. Per-rank estimates: " + str(memory_by_rank))
    optimizer = MainGradAdamW(model, args, opts.warmup_steps + opts.steps,
                             master_device=opts.optimizer_device)
    input_plan = json.loads(opts.input_plan.read_text()) if opts.input_plan else None
    if input_plan:
        check_plan_config(input_plan, opts.manifest, opts.batching, opts.microbatches,
                          opts.warmup_steps, opts.steps, opts.seq_length)
    batches = OptimusBatches(str(opts.model_dir), str(opts.manifest), os.environ["PROJ_ROOT"],
                             batching=opts.batching, input_plan=input_plan)
    empty_video = torch.zeros((0, 3), dtype=torch.long, device=device)
    profiles = {}
    fixed_owners = [m % world for m in range(opts.microbatches)]
    calibration_owners = fixed_owners if opts.encoder_placement == "round-robin" else [0] * opts.microbatches
    config = dict(vars(opts), implementation="optimus-inspired-pp-whole-encoder",
                  llm_parallelism={"pp": world, "tp": 1, "dp": 1, "layers_by_rank": layer_counts},
                  encoder_parallelism={"pp": 1, "tp": 1, "dp": world},
                  initialization="seeded-random; config/tokenizer from model-dir",
                  manifest_sha256=hashlib.sha256(opts.manifest.read_bytes()).hexdigest(),
                  input_plan_sha256=file_sha256(opts.input_plan) if input_plan else None,
                  sample_order_sha256=input_plan["sample_order_sha256"] if input_plan else None,
                  sample_multiset_sha256=input_plan["sample_multiset_sha256"] if input_plan else None,
                  timing_method=("synchronized phase wall time; diagnostic, changes overlap"
                                 if opts.profile_step_phases else "complete step wall time"),
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  memory_estimates=memory_by_rank,
                  optimizer={"name": "AdamW", "lr": args.lr, "min_lr": args.min_lr,
                             "betas": [args.adam_beta1, args.adam_beta2],
                             "eps": args.adam_eps, "weight_decay": args.weight_decay,
                             "clip_grad": args.clip_grad, "loss_reduction": "mean-of-microbatch-means"},
                  model_config_sha256=hashlib.sha256((opts.model_dir / "config.json").read_bytes()).hexdigest(),
                  code_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in ("optimus_qwen.py", "optimus_runtime.py", "optimus_schedule.py",
                                            "manifest_data.py", "optimus_optimizer.py", "optimus_config.py",
                                            "optimus_experiment.py", "optimus_profiling.py",
                                            "pp2_equivalence.py")},
                  pmp_commit=os.environ.get("PMP_COMMIT", "unrecorded"),
                  mcore_commit=os.environ.get("MCORE_COMMIT", "unrecorded"),
                  source_commit=os.environ.get("SOURCE_COMMIT", "unrecorded"),
                  unsupported=["kernel-level scheduling", "TP/DP bubble exploitation",
                               "encoder PP", "automatic 3D parallel search"])
    if rank == 0:
        (out / "run_config.json").write_text(json.dumps(config, default=str, indent=2) + "\n")
    encoder_trace = None
    if opts.trace_first_encoder_gradients:
        encoder_trace = {"initial_parameter": sample_encoder(model.vision_model, opts.validation_samples)}
    for step in range(opts.warmup_steps + opts.steps):
        data = batches.step_tensors(step, opts.microbatches, device,
                                    qwen.get_ltor_masks_and_position_ids, empty_video)
        if max(b["tokens"].shape[1] for b in data) > opts.seq_length:
            raise ValueError("processed sequence exceeds --seq-length")
        if input_plan:
            check_step_inputs(input_plan, step, data)
        adapter = QwenAdapter(model, data)
        executor = PlanExecutor(adapter, transport, device)
        signature = sample_signature(data)
        if signature not in profiles:
            # Profile a no-update reference pass on exactly these shapes. Clear
            # both fused and ordinary gradients before the actual training step.
            unit = {k: [[1.0] * opts.microbatches for _ in range(world)]
                    for k in ("EF", "EB", "F", "B")}
            calibration = make_plan(world, calibration_owners, unit, "serial")
            zero_grads(model)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                measured = executor.run(calibration)
            profiles[signature] = profile_costs(measured, world, calibration_owners)
        costs = profiles[signature]
        if opts.encoder_placement == "round-robin":
            plan = make_plan(world, fixed_owners, costs,
                             "serial" if opts.schedule == "serial-reference" else "bubble")
        else:
            plan = (make_plan(world, [0] * opts.microbatches, costs, "serial")
                    if opts.schedule == "serial-reference" else
                    choose_plan(world, opts.microbatches, costs))
        zero_grads(model)
        torch.cuda.reset_peak_memory_stats()
        dist.barrier()
        torch.cuda.synchronize()
        started = time.monotonic()
        profiler = StepProfiler(opts.profile_step_phases, synchronize=torch.cuda.synchronize)
        with profiler.phase("model_schedule"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = executor.run(plan, capture_image_grads=opts.validation_samples > 0)
        with profiler.phase("ordinary_grad_collect"):
            collect_main_grads(model)
        if step == 0 and encoder_trace is not None:
            encoder_trace["before_reduce"] = sample_encoder(
                model.vision_model, opts.validation_samples, gradients=True)
        with profiler.phase("encoder_grad_sync"):
            sync_encoder_grads(model.vision_model)
        if step == 0 and encoder_trace is not None:
            encoder_trace["after_reduce"] = sample_encoder(
                model.vision_model, opts.validation_samples, gradients=True)
        with profiler.phase("tied_grad_sync"):
            # Keep the pinned framework's tied-embedding synchronization.
            finalize_model_grads([model])
        with profiler.phase("global_grad_norm"):
            norm = global_grad_norm(model.vision_model, model.language_model)
        if not math.isfinite(result["loss"]) or not math.isfinite(norm):
            raise FloatingPointError("non-finite loss or gradient; optimizer was not advanced")
        with profiler.phase("optimizer"):
            stats = optimizer.step(global_grad_norm=norm, profiler=profiler)
        torch.cuda.synchronize()
        local = {"elapsed": time.monotonic() - started,
                 "peak": torch.cuda.max_memory_allocated()}
        if opts.profile_step_phases:
            local["phase_times_s"] = profiler.times
            local["phase_breakdown_s"] = profiler.breakdown(local["elapsed"])
        ranks = [None] * world
        dist.all_gather_object(ranks, local)
        record = dict(result, step=step, warmup=step < opts.warmup_steps,
                      samples=opts.microbatches, mode=opts.schedule, **stats,
                      step_time_s=max(r["elapsed"] for r in ranks),
                      max_memory_allocated_bytes_by_rank={str(r): v["peak"] for r, v in enumerate(ranks)},
                      sample_ids=[b["sample_id"] for b in data],
                      input_observations=[b["input_observation"] for b in data],
                      skipped=False, nan_or_inf=False,
                      optimizer_updates=optimizer.steps)
        if opts.profile_step_phases:
            critical = max(range(world), key=lambda r: ranks[r]["elapsed"])
            record.update(phase_times_s_by_rank={str(r): item["phase_times_s"]
                                                for r, item in enumerate(ranks)},
                          critical_rank=critical,
                          critical_rank_phase_breakdown_s=ranks[critical]["phase_breakdown_s"])
        if opts.validation_samples:
            sampled = {}
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                count = min(opts.validation_samples, param.numel())
                idx = sample_indices(param.numel(), count, device)
                sampled[name] = {"shape": list(param.shape),
                                 "parameter": param.detach().flatten()[idx].float().cpu().tolist(),
                                 "used_gradient": param.main_grad.flatten()[idx].float().cpu().tolist()}
            (out / ("validation-step%03d-rank%d.json" % (step, rank))).write_text(
                json.dumps(sampled) + "\n")
        if step == 0 and encoder_trace is not None:
            (out / ("encoder-trace-rank%d.json" % rank)).write_text(json.dumps(encoder_trace) + "\n")
        if rank == 0:
            (out / ("plan-%03d.json" % step)).write_text(json.dumps(plan, indent=2) + "\n")
            (out / ("profile-%03d.json" % step)).write_text(json.dumps(
                {"source": "CUDA calibration without optimizer update",
                 "unit": "ms", "costs": costs,
                 "encoder_calibration_owners": calibration_owners,
                 "encoder_cost_assumption": "owner measurements reused on identical GPU replicas"},
                indent=2) + "\n")
            with (out / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps({k: record[k] for k in ("step", "loss", "grad_norm", "step_time_s")}), flush=True)
    if rank == 0:
        (out / "status.json").write_text(json.dumps({"status": "completed", "steps": optimizer.steps}) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

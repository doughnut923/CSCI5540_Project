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

from manifest_data import ManifestBatches
from optimus_runtime import (PlanExecutor, Transport, collect_main_grads,
                             global_grad_norm, sample_indices, sync_encoder_grads, zero_grads)
from optimus_schedule import choose_plan, make_plan
from pp2_equivalence import mcore_argv
from training_common import MainGradAdamW


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


def build_model(qwen, args, rank, world, device):
    model = qwen.model_provider(pre_process=rank == 0, post_process=rank == world - 1)
    if model.vision_model is None:
        # Same constructor/configuration as the pinned team model_provider; only
        # placement differs. No duplicate language model is built on this rank.
        from megatron_patch.model.qwen2_vl.visionmodel import Qwen2VisionModel
        vc = qwen.get_vision_model_config(args, deepcopy(model.config))
        vc.pipeline_model_parallel_size = 1
        vc.first_pipeline_num_layers = None
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


def profile_costs(result, world, count):
    costs = {kind: [[None] * count for _ in range(world)] for kind in ("EF", "EB", "F", "B")}
    for iv in result["compute_intervals"]:
        # CUDA events cover model calls only, excluding explicit P2P waits.
        # These are whole-encoder/stage costs, not a kernel-level profile.
        costs[iv["kind"]][iv["rank"]][iv["microbatch"]] = max(
            (iv["end_s"] - iv["start_s"]) * 1000, 1e-6)
    for kind in ("EF", "EB"):
        for rank in range(1, world):
            costs[kind][rank] = list(costs[kind][0])
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
    p.add_argument("--memory-headroom-gb", type=float, default=2.0)
    p.add_argument("--optimizer-device", choices=("cuda", "cpu"), default="cuda",
                   help="CPU offloads FP32 master weights and AdamW states for the two-GPU quota")
    p.add_argument("--validation-samples", type=int, default=0,
                   help="Save evenly spaced parameter/used-gradient samples after timing")
    p.add_argument("--schedule", choices=("serial-reference", "optimus-pp"), default="optimus-pp")
    p.add_argument("--batching", choices=("bucket", "manifest"), default="manifest",
                   help="Manifest order preserves mixed sizes/counts within each step")
    opts = p.parse_args()
    if opts.steps < 1 or opts.warmup_steps < 0 or opts.microbatches < 1 or opts.validation_samples < 0:
        p.error("steps/microbatches must be positive; warmup must be nonnegative")
    if not math.isfinite(opts.memory_headroom_gb) or opts.memory_headroom_gb < 0:
        p.error("memory headroom must be finite and nonnegative")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world < 2 or opts.layers % world:
        p.error("requires PP >= 2 and a layer count divisible by PP")
    out = opts.output_dir.resolve()
    if int(os.environ.get("RANK", "0")) == 0:
        out.mkdir(parents=True, exist_ok=False)
    os.environ.update(QWEN2VL_HF_DIR=str(opts.model_dir), PP_SIZE=str(world),
                      NUM_LAYERS=str(opts.layers), NUM_MICROBATCHES=str(opts.microbatches))
    # The colocated executor owns scheduling; never install a second PP executor.
    os.environ.update(USE_PP_EXECUTOR="0", MEGATRON_USE_PP_EXECUTOR="0")
    sys.argv = ["optimus_qwen"] + mcore_argv(512)
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
    model = build_model(qwen, args, rank, world, device)
    transport = Transport(world, device)
    # AdamW needs FP32 masters and two moments in addition to the model and
    # main_grad already allocated. Check every rank before any starts training.
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    extra_bytes = parameter_count * 12 if opts.optimizer_device == "cuda" else 0
    free_bytes, _ = torch.cuda.mem_get_info()
    memory = {"rank": rank, "parameters": parameter_count,
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
    batches = ManifestBatches(str(opts.model_dir), str(opts.manifest), os.environ["PROJ_ROOT"],
                              batching=opts.batching)
    empty_video = torch.zeros((0, 3), dtype=torch.long, device=device)
    profiles = {}
    config = dict(vars(opts), implementation="optimus-inspired-pp-whole-encoder",
                  llm_parallelism={"pp": world, "tp": 1, "dp": 1},
                  encoder_parallelism={"pp": 1, "tp": 1, "dp": world},
                  initialization="seeded-random; config/tokenizer from model-dir",
                  manifest_sha256=hashlib.sha256(opts.manifest.read_bytes()).hexdigest(),
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  memory_estimates=memory_by_rank,
                  optimizer={"name": "AdamW", "lr": args.lr, "min_lr": args.min_lr,
                             "betas": [args.adam_beta1, args.adam_beta2],
                             "eps": args.adam_eps, "weight_decay": args.weight_decay,
                             "clip_grad": args.clip_grad, "loss_reduction": "mean-of-microbatch-means"},
                  model_config_sha256=hashlib.sha256((opts.model_dir / "config.json").read_bytes()).hexdigest(),
                  code_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in ("optimus_qwen.py", "optimus_runtime.py", "optimus_schedule.py",
                                            "manifest_data.py", "training_common.py", "pp2_equivalence.py")},
                  pmp_commit=os.environ.get("PMP_COMMIT", "unrecorded"),
                  mcore_commit=os.environ.get("MCORE_COMMIT", "unrecorded"),
                  source_commit=os.environ.get("SOURCE_COMMIT", "unrecorded"),
                  unsupported=["kernel-level scheduling", "TP/DP bubble exploitation",
                               "encoder PP", "automatic 3D parallel search"])
    if rank == 0:
        (out / "run_config.json").write_text(json.dumps(config, default=str, indent=2) + "\n")
    for step in range(opts.warmup_steps + opts.steps):
        data = batches.step_tensors(step, opts.microbatches, device,
                                    qwen.get_ltor_masks_and_position_ids, empty_video)
        adapter = QwenAdapter(model, data)
        executor = PlanExecutor(adapter, transport, device)
        signature = sample_signature(data)
        if signature not in profiles:
            # Profile a no-update reference pass on exactly these shapes. Clear
            # both fused and ordinary gradients before the actual training step.
            unit = {k: [[1.0] * opts.microbatches for _ in range(world)]
                    for k in ("EF", "EB", "F", "B")}
            calibration = make_plan(world, [0] * opts.microbatches, unit, "serial")
            zero_grads(model)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                measured = executor.run(calibration)
            profiles[signature] = profile_costs(measured, world, opts.microbatches)
        costs = profiles[signature]
        plan = (make_plan(world, [0] * opts.microbatches, costs, "serial")
                if opts.schedule == "serial-reference" else
                choose_plan(world, opts.microbatches, costs))
        zero_grads(model)
        torch.cuda.reset_peak_memory_stats()
        dist.barrier()
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = executor.run(plan, capture_image_grads=opts.validation_samples > 0)
        collect_main_grads(model)
        sync_encoder_grads(model.vision_model)
        # Retain the pinned framework's tied-embedding gradient synchronization.
        finalize_model_grads([model])
        norm = global_grad_norm(model.vision_model, model.language_model)
        if not math.isfinite(result["loss"]) or not math.isfinite(norm):
            raise FloatingPointError("non-finite loss or gradient; optimizer was not advanced")
        stats = optimizer.step(global_grad_norm=norm)
        torch.cuda.synchronize()
        local = {"elapsed": time.monotonic() - started,
                 "peak": torch.cuda.max_memory_allocated()}
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
        if rank == 0:
            (out / ("plan-%03d.json" % step)).write_text(json.dumps(plan, indent=2) + "\n")
            (out / ("profile-%03d.json" % step)).write_text(json.dumps(
                {"source": "CUDA calibration without optimizer update",
                 "unit": "ms", "costs": costs,
                 "encoder_cost_assumption": "rank-0 measurements reused on identical GPU replicas"},
                indent=2) + "\n")
            with (out / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps({k: record[k] for k in ("step", "loss", "grad_norm", "step_time_s")}), flush=True)
    if rank == 0:
        (out / "status.json").write_text(json.dumps({"status": "completed", "steps": optimizer.steps}) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

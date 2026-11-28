"""Part B (reduced DistTrain): disaggregated vision encoder + LLM pipeline.

Two independent torchrun worlds on one node, launched concurrently by
slurm/disagg_units.sbatch:

  enc  ranks 0..E-1, DP=E        : vision tower only, one microbatch per rank
  llm  ranks 0..P-1, PP=P, DP=1  : LLM-only pipeline, no ViT and no vision scatter

They exchange per-microbatch vision embeddings and their gradients through
/dev/shm and lock-step through flag files. Weights are seeded CPU init (no
checkpoint). Both units step AdamW over the fused fp32 `main_grad` buffers each
step, so the loss curve is a real one; the encoder unit only steps when its
gradients are complete (world 1, or ENC_GRAD_SYNC all-reduce), and otherwise the
run stays placement-only.

Microbatches come from the pilot manifest via ManifestBatches: step s draws its
microbatches from one image-token group, so a step has a single sequence length.
PLAN_ONLY=1 prints the plan and the per-step sequence length without touching GPUs.

Every rank records CUDA-event intervals: "enc" (vision tower), "llm" (language
model forward), "bwd" (backward). The LLM rank 0 merges both units into one JSONL
record for scripts/plot_pipeline_gantt.py.

Env: ROLE=enc|llm, DISAGG_SHM, TAG, NUM_MICROBATCHES, LLM_PP, BENCH_STEPS,
WARMUP_STEPS, NUM_LAYERS, PROJ_ROOT, MANIFEST, QWEN2VL_HF_DIR, PLAN_ONLY.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest_data import ManifestBatches  # noqa: E402  (same-directory helper)


def env_int(name, default):
    return int(os.environ.get(name, str(default)))


def mcore_argv(num_layers, pp_size, num_microbatches):
    argv = [
        "--num-layers", str(num_layers),
        "--hidden-size", "1536",
        "--ffn-hidden-size", "8960",
        "--num-attention-heads", "12",
        "--num-query-groups", "2",
        "--group-query-attention",
        "--max-position-embeddings", "32768",
        # 1024: the Part C image-count cases are 607-621 tokens, above the old 512.
        "--seq-length", "1024",
        "--micro-batch-size", "1",
        "--global-batch-size", str(num_microbatches),
        "--pipeline-model-parallel-size", str(pp_size),
        "--tensor-model-parallel-size", "1",
        "--use-mcore-models",
        "--swiglu",
        "--disable-bias-linear",
        "--add-qkv-bias",
        "--normalization", "RMSNorm",
        "--norm-epsilon", "1e-6",
        "--rotary-base", "1000000",
        "--use-rotary-position-embeddings",
        "--no-rope-fusion",
        "--no-bias-swiglu-fusion",
        "--attention-dropout", "0.0",
        "--hidden-dropout", "0.0",
        "--bf16",
        "--use-cpu-initialization",
        "--seed", "1234",
        "--train-iters", "1",
        "--eval-interval", "1000000",
        "--eval-iters", "0",
        "--save-interval", "1000000",
        "--lr", "1e-4",
        "--min-lr", "1e-5",
        "--lr-decay-iters", "1",
        "--lr-decay-style", "cosine",
        "--weight-decay", "0.01",
        "--clip-grad", "1.0",
        "--adam-beta1", "0.9",
        "--adam-beta2", "0.95",
        "--transformer-impl", "transformer_engine",
        "--patch-tokenizer-type", "Qwen2Tokenizer",
        "--extra-vocab-size", "293",
        "--spatial-merge-size", "2",
        "--load", os.environ["QWEN2VL_HF_DIR"],
        "--save", "/tmp/disagg_unused",
    ]
    # Uneven pipeline split for depths that do not divide the layer count, e.g. 28 over
    # 6 stages as 4|5|5|5|5|4. Megatron's default split asserts on divisibility; these two
    # flags take the first/last stage out of the even split and divide the rest. The
    # encoder unit runs a one-layer, one-stage model, so it must not receive them.
    first = os.environ.get("PP_FIRST_LAYERS")
    last = os.environ.get("PP_LAST_LAYERS")
    if pp_size > 1 and (first or last):
        if first:
            argv += ["--decoder-first-pipeline-num-layers", first]
        if last:
            argv += ["--decoder-last-pipeline-num-layers", last]
    return argv


class Shm:
    """Flag + payload exchange between the two units on one node."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def touch(self, name):
        (self.root / name).write_text("")

    def wait(self, name, timeout=900.0):
        p = self.root / name
        t0 = time.monotonic()
        while not p.exists():
            if time.monotonic() - t0 > timeout:
                raise RuntimeError("timeout waiting for %s (%.0fs)" % (p, timeout))
            time.sleep(0.003)

    def save(self, name, obj):
        torch.save(obj, self.root / name)

    def load(self, name):
        return torch.load(self.root / name, map_location="cpu", weights_only=False)

    def write_json(self, name, obj):
        (self.root / name).write_text(json.dumps(obj))

    def read_json(self, name):
        return json.loads((self.root / name).read_text())


class MainGradAdamW:
    """AdamW over MCore's fp32 main_grad buffers, with fp32 master weights.

    The fused path accumulates into `param.main_grad` and never writes `.grad`, so the
    fp32 masters take those buffers as their grads and are cast back into the bf16 params
    after every update (the same split Megatron's mixed-precision optimizer uses).
    """

    def __init__(self, model, args, total_steps):
        self.params = [p for p in model.parameters()
                       if getattr(p, "main_grad", None) is not None]
        self.masters = [p.detach().float().clone().requires_grad_(True) for p in self.params]
        self.opt = torch.optim.AdamW(
            self.masters, lr=args.lr, betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps, weight_decay=args.weight_decay)
        self.lr, self.min_lr = args.lr, args.min_lr
        self.clip = args.clip_grad
        self.total = max(1, total_steps)
        self.steps = 0

    def step(self):
        for master, p in zip(self.masters, self.params):
            master.grad = p.main_grad
        lr = self.min_lr + 0.5 * (self.lr - self.min_lr) * (
            1.0 + math.cos(math.pi * self.steps / self.total))
        for group in self.opt.param_groups:
            group["lr"] = lr
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.masters, self.clip))
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for master, p in zip(self.masters, self.params):
                p.data.copy_(master)
        self.steps += 1
        return {"lr": lr, "grad_norm": grad_norm,
                "weight_norm": float(self.masters[0].norm())}


def main():
    num_mb = env_int("NUM_MICROBATCHES", 4)
    batches = ManifestBatches(os.environ["QWEN2VL_HF_DIR"], os.environ["MANIFEST"],
                              os.environ["PROJ_ROOT"],
                              batching=os.environ.get("MANIFEST_BATCHING", "bucket"))
    if env_int("PLAN_ONLY", 0):
        for step in range(env_int("PLAN_STEPS", 4)):
            built = batches.build(step, num_mb)
            print(json.dumps({"step": step,
                              "sample_ids": [b["sample_id"] for b in built],
                              "seq_length": built[0]["tokens"].shape[1]}), flush=True)
        return

    role = os.environ["ROLE"]
    tag = os.environ.get("TAG", "v1")
    shm = Shm(os.environ["DISAGG_SHM"])

    llm_pp = env_int("LLM_PP", 4)
    bench_steps = env_int("BENCH_STEPS", 3)
    warmup_steps = env_int("WARMUP_STEPS", 2)
    num_layers = env_int("NUM_LAYERS", 32)
    # torchrun has not initialised torch.distributed yet; Megatron does that later.
    # torchrun sets these; torch.distributed itself is initialised by Megatron below.
    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    if role == "enc" and num_mb % world:
        # The encoder world is a plain data-parallel group: Megatron rejects a global
        # batch it cannot split evenly, and the failure surfaces 15 min later as a
        # ready_enc timeout on the other unit. Fail at the source instead.
        raise SystemExit("NUM_MICROBATCHES (%d) must be a multiple of ENC_GPUS (%d)"
                         % (num_mb, world))

    mcore_layers = 1 if role == "enc" else num_layers
    mcore_pp = 1 if role == "enc" else llm_pp
    sys.argv = ["disagg_" + role] + mcore_argv(mcore_layers, mcore_pp, num_mb)
    from megatron.training.initialize import initialize_megatron
    from megatron_patch.arguments import get_patch_args
    initialize_megatron(extra_args_provider=get_patch_args)

    from megatron.training import get_args
    from qwen2_vl.pretrain_qwen import get_ltor_masks_and_position_ids, model_provider

    args = get_args()
    device = torch.device("cuda", torch.cuda.current_device())
    empty_video = torch.zeros((0, 3), dtype=torch.long, device=device)

    def step_batches(step):
        return batches.step_tensors(step, num_mb, device,
                                    get_ltor_masks_and_position_ids, empty_video)

    def ev():
        return torch.cuda.Event(enable_timing=True)

    def timed(fn):
        """Return (result, (cpu_start, cuda_start_event, cuda_end_event))."""
        start, end = ev(), ev()
        cpu = time.monotonic()
        start.record()
        result = fn()
        end.record()
        return result, (cpu, start, end)

    if role == "enc":
        model = model_provider(pre_process=True, post_process=True)
        vision = model.vision_model
        model.language_model = None  # only the tower is trained in this unit
        vision = vision.to(device).train()
        for p in vision.parameters():
            if p.requires_grad and getattr(p, "main_grad", None) is None:
                p.main_grad = torch.zeros_like(p, dtype=torch.float32)
        shm.touch("ready_enc")
    else:
        model = model_provider(pre_process=(rank == 0), post_process=(rank == llm_pp - 1))
        if rank == 0:
            model.vision_model = None  # the tower lives in the other unit
        model = model.to(device).train()
        from contextlib import nullcontext
        from megatron.core.enums import ModelType
        from megatron.core.distributed import DistributedDataParallelConfig
        from megatron.core.pipeline_parallel import get_forward_backward_func
        import megatron.core.pipeline_parallel.schedules as _sched
        model.model_type = ModelType.encoder_or_decoder
        model.finish_grad_sync = lambda: None
        model.start_grad_sync = lambda: None
        model.no_sync = lambda: nullcontext()
        model.ddp_config = DistributedDataParallelConfig()
        for p in model.parameters():
            if p.requires_grad and getattr(p, "main_grad", None) is None:
                p.main_grad = torch.zeros_like(p, dtype=torch.float32)
        # keeps MCore on torch.autograd.backward, which the backward timer wraps
        model.config.deallocate_pipeline_outputs = False
        if hasattr(_sched, "finalize_model_grads"):
            _sched.finalize_model_grads = lambda *a, **k: None
        fwd_bwd = get_forward_backward_func()
        shm.touch("ready_llm")

    step_opt = MainGradAdamW(model, args, warmup_steps + bench_steps)

    # Fingerprint of the initial weights, before any optimizer step. Equal fingerprints
    # across arms mean any loss difference comes from the computation, not the init.
    _norm0 = torch.zeros((), device=device)
    for p in model.parameters():
        _norm0 += p.detach().float().pow(2).sum()
    def _module_norms(module):
        out = {}
        if module is not None:
            for name, child in module.named_children():
                total = torch.zeros((), device=device)
                for p in child.parameters():
                    total += p.detach().float().pow(2).sum()
                out[name] = float(total)
        return out

    param_norm0 = {"total": float(_norm0), "modules": _module_norms(model)}

    shm.wait("ready_enc")
    shm.wait("ready_llm")
    torch.distributed.barrier()

    records = []
    for it in range(warmup_steps + bench_steps):
        measured = it >= warmup_steps
        step_data = step_batches(it)
        if it + 1 == warmup_steps:  # peak memory should describe the measured steps
            torch.cuda.reset_peak_memory_stats()
        if rank == 0:
            print("step %d %s samples %s" % (it, role, [b["sample_id"] for b in step_data]),
                  flush=True)
        # llm rank 0 publishes the step t0 (CLOCK_MONOTONIC is host-wide).
        if role == "llm" and rank == 0:
            shm.write_json("cmd_%d.json" % it, {"t0": time.monotonic()})
            shm.touch("cmd_%d.ok" % it)
        shm.wait("cmd_%d.ok" % it)
        t0 = shm.read_json("cmd_%d.json" % it)["t0"]

        raw = []  # (unit_rank, kind, cpu_start, cuda_start_event, cuda_end_event)

        if role == "enc":
            my_mbs = list(range(rank, num_mb, world))
            embs = {}
            for p in vision.parameters():
                if getattr(p, "main_grad", None) is not None:
                    p.main_grad.zero_()
                elif p.grad is not None:
                    p.grad = None
            for mb in my_mbs:
                b = step_data[mb]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    emb, t_fwd = timed(
                        lambda: vision(vision_data=b["pixels"], grid_thw=b["grid"]))
                raw.append((rank, "enc") + t_fwd)
                shm.save("emb_%d_%d.pt" % (it, mb), emb.detach().float().cpu())
                shm.touch("emb_%d_%d.ok" % (it, mb))
                embs[mb] = emb

            image_grads = {}
            for mb in my_mbs:
                shm.wait("gemb_%d_%d.ok" % (it, mb))
                grad = shm.load("gemb_%d_%d.pt" % (it, mb)).to(
                    device=device, dtype=torch.bfloat16)
                _, t_bwd = timed(lambda: embs[mb].backward(grad))
                raw.append((rank, "bwd") + t_bwd)
                # Per-image slice of the returned embedding gradient: proves each image
                # reached the model, not just the processor (Part C's open measurement).
                counts = (step_data[mb].get("input_observation") or {}).get(
                    "visual_tokens_per_image")
                if counts:
                    if sum(counts) != grad.shape[0]:
                        raise RuntimeError("image slices %s do not sum to the %d embedding rows"
                                           % (counts, grad.shape[0]))
                    offset, norms = 0, []
                    for count in counts:
                        norms.append(float(grad[offset:offset + count].float().norm()))
                        offset += count
                    image_grads[step_data[mb]["sample_id"]] = norms

            # Real encoder DP averages gradients across the replicas; that cost is part
            # of the step, so measure it rather than hide it. World 1 means no DP.
            if world > 1 and env_int("ENC_GRAD_SYNC", 1):
                def _all_reduce_grads():
                    for p in vision.parameters():
                        g = getattr(p, "main_grad", None)
                        if g is not None:
                            torch.distributed.all_reduce(
                                g, op=torch.distributed.ReduceOp.SUM)
                _, t_sync = timed(_all_reduce_grads)
                raw.append((rank, "sync") + t_sync)

            if world == 1 or env_int("ENC_GRAD_SYNC", 1):
                def _update():
                    if world > 1:  # the all-reduce summed, so average before stepping
                        for p in vision.parameters():
                            g = getattr(p, "main_grad", None)
                            if g is not None:
                                g.div_(world)
                    return step_opt.step()
                opt_stats, t_opt = timed(_update)
                raw.append((rank, "opt") + t_opt)
            else:
                # Unsynchronised replicas only hold their own microbatches' gradients:
                # stepping on them would be wrong, so this config stays placement-only.
                opt_stats = None

            torch.cuda.synchronize()
            step_time = time.monotonic() - t0
            merged = []
            for r, k, cpu, s, e in raw:
                off = cpu - t0
                merged.append([r, k, off, off + s.elapsed_time(e) / 1000.0])
            gathered = [None] * world
            torch.distributed.all_gather_object(
                gathered, {"intervals": merged, "step_time_s": step_time,
                           "opt": opt_stats, "peak": torch.cuda.max_memory_allocated(),
                           "param_norm0": param_norm0, "image_grads": image_grads})
            torch.distributed.barrier()
            if rank == 0:
                flat = []
                worst = 0.0
                for chunk in gathered:
                    flat.extend(chunk["intervals"])
                    worst = max(worst, chunk["step_time_s"])
                image_grad_norms = {}
                for chunk in gathered:
                    image_grad_norms.update(chunk.get("image_grads") or {})
                shm.write_json("iv_%d.json" % it,
                               {"intervals": flat, "world": world, "step_time_s": worst,
                                "opt": gathered[0]["opt"],
                                "image_grad_norms": image_grad_norms,
                                "param_norm0_by_rank": {str(r): gathered[r]["param_norm0"]["total"]
                                                        for r in range(world)},
                                "module_norm0_by_rank": {str(r): gathered[r]["param_norm0"]["modules"]
                                                         for r in range(world)},
                                "peak_by_rank": {str(r): int(gathered[r]["peak"])
                                                 for r in range(world)}})
                shm.touch("enc_done_%d.ok" % it)
        else:
            pending = {}
            pending = {}
            emb_leaves = {}

            def _forward(data_iterator, model):
                mb = next(data_iterator)["mb"]
                b = step_data[mb]
                if rank == 0:
                    if mb not in pending:  # start the pipeline as soon as this mb lands
                        shm.wait("emb_%d_%d.ok" % (it, mb))
                        pending[mb] = shm.load("emb_%d_%d.pt" % (it, mb))
                    leaf = pending[mb].to(device=device, dtype=torch.bfloat16)
                    leaf.requires_grad_(True)
                    emb_leaves[mb] = leaf

                    def _run():
                        combined = model.language_model.embedding(
                            input_ids=b["tokens"], position_ids=None,
                            image_input_mask=b["image_mask"].T, video_input_mask=None,
                            image_embeds=leaf, video_embeds=None)
                        return model.language_model(
                            input_ids=None, position_ids=b["position_ids"],
                            attention_mask=None, decoder_input=combined,
                            labels=b["target"])
                else:
                    def _run():
                        return model(
                            input_ids=b["tokens"], position_ids=b["position_ids"],
                            vision_data=b["pixels"], vision_grid_thw=b["grid"],
                            video_start_index=int(b["image_mask"].sum().item()),
                            image_input_mask=b["image_mask"], video_input_mask=None,
                            attention_mask=None, labels=b["target"])

                out, t = timed(_run)
                raw.append((rank, "llm") + t)

                def loss_func(output_tensor):
                    mask = b["loss_mask"].view(-1).float()
                    loss = (output_tensor.float().view(-1) * mask).sum() / mask.sum()
                    return loss, {"loss": loss.detach()}

                return out, loss_func

            def _mb_iter():
                while True:
                    for mb in range(num_mb):
                        yield {"mb": mb}

            import torch.autograd as _ag
            _orig_backward = _ag.backward
            bwd_count = [0]

            def _timed_backward(tensors, grad_tensors=None, **kw):
                _, t = timed(lambda: _orig_backward(
                    tensors, grad_tensors=grad_tensors, **kw))
                raw.append((rank, "bwd") + t)
                # Stage 0 finishes the backward of each microbatch in order; hand its
                # embedding gradient to the encoder unit immediately so the encoder's
                # backward overlaps the rest of the LLM pipeline.
                if rank == 0 and bwd_count[0] < num_mb:
                    mb = bwd_count[0]
                    bwd_count[0] += 1
                    leaf = emb_leaves.get(mb)
                    g = None if leaf is None else leaf.grad
                    if g is None:
                        g = torch.zeros(1)
                    shm.save("gemb_%d_%d.pt" % (it, mb), g.float().cpu())
                    shm.touch("gemb_%d_%d.ok" % (it, mb))

            _ag.backward = _timed_backward
            for p in model.parameters():
                if getattr(p, "main_grad", None) is not None:
                    p.main_grad.zero_()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = fwd_bwd(
                    forward_step_func=_forward,
                    data_iterator=_mb_iter(),
                    model=[model],
                    num_microbatches=num_mb,
                    seq_length=max(b["tokens"].shape[1] for b in step_data),
                    micro_batch_size=1,
                    forward_only=False,
                )
            _ag.backward = _orig_backward

            opt_stats, t_opt = timed(step_opt.step)
            raw.append((rank, "opt") + t_opt)
            # Per-stage params are disjoint, so squared norms add for a whole-model norm.
            norm_sq = torch.tensor([opt_stats["grad_norm"] ** 2], device=device)
            torch.distributed.all_reduce(norm_sq)

            values = []
            for item in out or []:
                val = item.get("loss") if isinstance(item, dict) else item
                if val is not None:
                    values.append(float(val))
            mb_losses = torch.full((num_mb,), float("nan"), device=device)
            for i, val in enumerate(values):
                mb_losses[i] = float(val)
            torch.distributed.broadcast(mb_losses, src=llm_pp - 1)
            mb_list = mb_losses.tolist()
            step_loss = float(sum(mb_list) / len(mb_list))

            torch.cuda.synchronize()
            step_time = time.monotonic() - t0
            merged = []
            for r, k, cpu, s, e in raw:
                off = cpu - t0
                merged.append([r, k, off, off + s.elapsed_time(e) / 1000.0])
            gathered = [None] * world
            torch.distributed.all_gather_object(
                gathered, {"intervals": merged, "step_time_s": step_time,
                           "peak": torch.cuda.max_memory_allocated(),
                           "param_norm0": param_norm0})
            if rank == 0:
                shm.wait("enc_done_%d.ok" % it)
                enc_side = shm.read_json("iv_%d.json" % it)
                base = enc_side["world"]
                flat = [[r, k, s, e] for r, k, s, e in enc_side["intervals"]]
                worst = enc_side["step_time_s"]
                for chunk in gathered:
                    flat.extend([[base + r, k, s, e]
                                 for r, k, s, e in chunk["intervals"]])
                    worst = max(worst, chunk["step_time_s"])
                peak_by_rank = {str(base + r): int(chunk["peak"])
                                for r, chunk in enumerate(gathered)}
                for r, v in (enc_side.get("peak_by_rank") or {}).items():
                    peak_by_rank[str(int(r))] = v
                param_norm0_by_rank = {str(base + r): chunk["param_norm0"]["total"]
                                       for r, chunk in enumerate(gathered)}
                module_norm0_by_rank = {str(base + r): chunk["param_norm0"]["modules"]
                                        for r, chunk in enumerate(gathered)}
                for r, v in (enc_side.get("param_norm0_by_rank") or {}).items():
                    param_norm0_by_rank[str(int(r))] = v
                for r, v in (enc_side.get("module_norm0_by_rank") or {}).items():
                    module_norm0_by_rank[str(int(r))] = v
                busy_total = sum(e - s for _, _, s, e in flat)
                capacity = (base + world) * worst
                bubble_time = max(0.0, capacity - busy_total)
                if measured:
                    records.append({
                        "step": it - warmup_steps,
                        "warmup": False,
                        "step_time_s": worst,
                        "samples": num_mb,
                        "max_memory_allocated_bytes_by_rank": peak_by_rank,
                        "param_norm0_by_rank": param_norm0_by_rank,
                        "module_norm0_by_rank": module_norm0_by_rank,
                        "busy_time_s_total": busy_total,
                        "bubble_time_s": bubble_time,
                        "pipeline_bubble_ratio": (bubble_time / capacity
                                                  if capacity > 0 else None),
                        "busy_intervals": flat,
                        "rank_labels": dict(
                            [(str(r), "enc %d" % r) for r in range(base)] +
                            [(str(base + r), "LLM %d" % r) for r in range(world)]),
                        "loss": step_loss,
                        "microbatch_losses": mb_list,
                        "sample_ids": [b["sample_id"] for b in step_data],
                        "input_observations": [b["input_observation"] for b in step_data],
                        "image_embedding_grad_norms": enc_side.get("image_grad_norms") or {},
                        "lr": opt_stats["lr"],
                        "grad_norm": opt_stats["grad_norm"],
                        "grad_norm_total": math.sqrt(
                            float(norm_sq) + ((enc_side.get("opt") or {})
                                              .get("grad_norm") or 0.0) ** 2),
                        "weight_norm": opt_stats["weight_norm"],
                        "enc_grad_norm": (enc_side.get("opt") or {}).get("grad_norm"),
                        "enc_weight_norm": (enc_side.get("opt") or {}).get("weight_norm"),
                        "skipped": not math.isfinite(step_loss),
                        "nan_or_inf": not math.isfinite(step_loss),
                        "pp_size": llm_pp,
                        "enc_world": base,
                        "num_layers": num_layers,
                        "num_microbatches": num_mb,
                        "mode": "disaggregated",
                    })
            torch.distributed.barrier()
            if rank == 0:
                shm.touch("llm_done_%d.ok" % it)

        if role == "enc":
            shm.wait("llm_done_%d.ok" % it)

    if role == "llm" and rank == 0:
        out_dir = Path(os.environ.get("OUT_DIR")
                       or Path(os.environ["PROJ_ROOT"]) / "runs" /
                       "track_b_n2_pilot_measured_v1")
        out_path = out_dir / ("disagg_m%d_%s.jsonl" % (num_mb, tag))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("".join(json.dumps(r) + "\n" for r in records))
        print(json.dumps({"wrote": str(out_path), "records": len(records)}), flush=True)


if __name__ == "__main__":
    main()

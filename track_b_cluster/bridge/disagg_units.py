"""Part B (reduced DistTrain): disaggregated vision encoder + LLM pipeline.

Two independent torchrun worlds on one node, launched concurrently by
slurm/disagg_units.sbatch:

  enc  ranks 0..E-1, DP=E        : vision tower only, one microbatch per rank
  llm  ranks 0..P-1, PP=P, DP=1  : LLM-only pipeline, no ViT and no vision scatter

They exchange per-microbatch vision embeddings and their gradients through
/dev/shm and lock-step through flag files. Weights are seeded CPU init (no
checkpoint) and encoder weight-gradients are all-reduced only when ENC_GRAD_SYNC
is set and the encoder world is > 1, so the sync-free runs measure placement and
schedule, not training fidelity.

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
from pp2_equivalence import ManifestBatches  # noqa: E402  (same-directory helper)


def env_int(name, default):
    return int(os.environ.get(name, str(default)))


def mcore_argv(num_layers, pp_size, num_microbatches):
    return [
        "--num-layers", str(num_layers),
        "--hidden-size", "1536",
        "--ffn-hidden-size", "8960",
        "--num-attention-heads", "12",
        "--num-query-groups", "2",
        "--group-query-attention",
        "--max-position-embeddings", "32768",
        "--seq-length", "512",
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


def main():
    num_mb = env_int("NUM_MICROBATCHES", 4)
    batches = ManifestBatches(os.environ["QWEN2VL_HF_DIR"], os.environ["MANIFEST"],
                              os.environ["PROJ_ROOT"])
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
        """One step's microbatches on device, with next-token targets for MCore.

        Built before the step's start time is published, so host preprocessing and the
        host-to-device copies stay out of the measured step time.
        """
        prepared = []
        for b in batches.build(step, num_mb):
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            target = b["labels"].clone()
            target[:, :-1] = b["labels"][:, 1:]
            target[:, -1] = -100
            _, b["loss_mask"], b["position_ids"] = get_ltor_masks_and_position_ids(
                b["tokens"], b["grid"], empty_video, target, 0, ignore_index=-100)
            b["target"] = target
            prepared.append(b)
        return prepared

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

    shm.wait("ready_enc")
    shm.wait("ready_llm")
    torch.distributed.barrier()

    records = []
    for it in range(warmup_steps + bench_steps):
        measured = it >= warmup_steps
        step_data = step_batches(it)
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

            for mb in my_mbs:
                shm.wait("gemb_%d_%d.ok" % (it, mb))
                grad = shm.load("gemb_%d_%d.pt" % (it, mb)).to(
                    device=device, dtype=torch.bfloat16)
                _, t_bwd = timed(lambda: embs[mb].backward(grad))
                raw.append((rank, "bwd") + t_bwd)

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

            torch.cuda.synchronize()
            step_time = time.monotonic() - t0
            merged = []
            for r, k, cpu, s, e in raw:
                off = cpu - t0
                merged.append([r, k, off, off + s.elapsed_time(e) / 1000.0])
            gathered = [None] * world
            torch.distributed.all_gather_object(
                gathered, {"intervals": merged, "step_time_s": step_time})
            torch.distributed.barrier()
            if rank == 0:
                flat = []
                worst = 0.0
                for chunk in gathered:
                    flat.extend(chunk["intervals"])
                    worst = max(worst, chunk["step_time_s"])
                shm.write_json("iv_%d.json" % it,
                               {"intervals": flat, "world": world, "step_time_s": worst})
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
                    seq_length=step_data[0]["tokens"].shape[1],
                    micro_batch_size=1,
                    forward_only=False,
                )
            _ag.backward = _orig_backward

            values = []
            for item in out or []:
                val = item.get("loss") if isinstance(item, dict) else item
                if val is not None:
                    values.append(float(val))
            local_loss = sum(values) / len(values) if values else float("nan")
            loss_t = torch.tensor([local_loss], device=device)
            torch.distributed.broadcast(loss_t, src=llm_pp - 1)
            step_loss = float(loss_t.item())

            torch.cuda.synchronize()
            step_time = time.monotonic() - t0
            merged = []
            for r, k, cpu, s, e in raw:
                off = cpu - t0
                merged.append([r, k, off, off + s.elapsed_time(e) / 1000.0])
            gathered = [None] * world
            torch.distributed.all_gather_object(
                gathered, {"intervals": merged, "step_time_s": step_time})
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
                if measured:
                    records.append({
                        "step": it - warmup_steps,
                        "warmup": False,
                        "step_time_s": worst,
                        "samples": num_mb,
                        "busy_intervals": flat,
                        "rank_labels": dict(
                            [(str(r), "enc %d" % r) for r in range(base)] +
                            [(str(base + r), "LLM %d" % r) for r in range(world)]),
                        "loss": step_loss,
                        "sample_ids": [b["sample_id"] for b in step_data],
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
        out_path = (Path(os.environ["PROJ_ROOT"]) / "runs" /
                    "track_b_n2_pilot_measured_v1" / ("disagg_m%d_%s.jsonl" % (num_mb, tag)))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("".join(json.dumps(r) + "\n" for r in records))
        print(json.dumps({"wrote": str(out_path), "records": len(records)}), flush=True)


if __name__ == "__main__":
    main()

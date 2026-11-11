"""PP equivalence check: native MCore pipeline schedule vs BigMac pp_executor.

Both runs build the same model (identical seed, CPU init), consume the same sample, and
execute the same two microbatches. Only the forward-backward function differs, so loss and
gradient norm must match. This is the G2 gate before any timing is collected.

PP size and layer count come from PP_SIZE / NUM_LAYERS (default 2 / 28, the proven G2
config). PP=8 cannot split 28 layers evenly, so scaling runs set NUM_LAYERS=32; weights
are seeded CPU init either way, so this stays a schedule-scaling test, not a fidelity run.

Benchmark mode (BENCH_STEPS > 0) repeats the step and records, per step:
  step_time_s, samples, peak allocated bytes per rank, and the exact pipeline bubble
  (bubble_time_s = pp_size * step_time - sum of per-rank compute-busy time). See
  docs/pipeline-bubble.md for the definition and its limits.

Encoder nesting (LLMPPExecutorWithVE) is NOT exercised here; this is the text-side schedule.
"""

import json
import math
import os
import sys
from pathlib import Path

import torch


def env_int(name, default):
    return int(os.environ.get(name, str(default)))


def mcore_argv(seq_length):
    num_layers = os.environ.get("NUM_LAYERS", "28")
    pp_size = os.environ.get("PP_SIZE", "2")
    num_microbatches = os.environ.get("NUM_MICROBATCHES", "2")
    return [
        "--num-layers", num_layers,
        "--hidden-size", "1536",
        "--ffn-hidden-size", "8960",
        "--num-attention-heads", "12",
        "--num-query-groups", "2",
        "--group-query-attention",
        "--max-position-embeddings", "32768",
        "--seq-length", str(seq_length),
        "--micro-batch-size", "1",
        # DP=1, so global batch == microbatch * microbatches.
        "--global-batch-size", num_microbatches,
        "--pipeline-model-parallel-size", pp_size,
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
        "--save", "/tmp/pp2_unused",
    ]


def build_batch(hf_dir, manifest_path, repo_root, device):
    import json as _json
    from PIL import Image
    from transformers import AutoProcessor

    manifest = _json.loads(Path(manifest_path).read_text())
    samples = manifest["samples"]
    sample = next((s for s in samples if s["images"][0]["resolution"] == 128), samples[0])
    images = []
    for img in sample["images"]:
        path = Path(img["path_relative_to_repo_root"])
        if not path.is_absolute():
            path = Path(repo_root) / path
        with Image.open(path) as src:
            res = int(img["resolution"])
            images.append(src.convert("RGB").resize((res, res)))

    processor = AutoProcessor.from_pretrained(hf_dir, local_files_only=True, use_fast=False)
    message = {"role": "user", "content": [{"type": "image"} for _ in images]
               + [{"type": "text", "text": "Describe these images."}]}
    prefix = processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
    full = processor.apply_chat_template(
        [message, {"role": "assistant", "content": [{"type": "text", "text": "Two images."}]}],
        tokenize=False, add_generation_prompt=False)
    batch = dict(processor(text=[full], images=images, padding=True, return_tensors="pt"))
    prefix_len = processor(text=[prefix], images=images, padding=True,
                           return_tensors="pt")["input_ids"].shape[1]

    labels = batch["input_ids"].clone()
    labels[:, :prefix_len] = -100
    labels[batch["attention_mask"] == 0] = -100

    tokens = batch["input_ids"].to(device)
    grid = batch["image_grid_thw"].to(device)
    pixels = batch["pixel_values"].to(device, dtype=torch.bfloat16)
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    image_mask = tokens == image_token_id
    return tokens, labels.to(device), grid, pixels, image_mask, sample["sample_id"]


def main():
    out_path = Path(sys.argv[1])
    mode = os.environ.get("USE_PP_EXECUTOR", "0")
    repo_root = Path(os.environ["PROJ_ROOT"])
    hf_dir = os.environ["QWEN2VL_HF_DIR"]
    pp_size_env = os.environ.get("PP_SIZE", "2")
    num_layers = os.environ.get("NUM_LAYERS", "28")
    bench_steps = env_int("BENCH_STEPS", 0)
    warmup_steps = env_int("WARMUP_STEPS", 2)
    num_microbatches = env_int("NUM_MICROBATCHES", 2)

    # Must happen before megatron initializes, so the patch is in place on import.
    if mode in ("1", "true", "yes", "on"):
        from pp_executor.mcore_patch import install
        install()

    sys.argv = ["pp2_equiv"] + mcore_argv(512)
    from megatron.training.initialize import initialize_megatron
    from megatron_patch.arguments import get_patch_args
    initialize_megatron(extra_args_provider=get_patch_args)

    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from megatron.training import get_args
    from qwen2_vl.pretrain_qwen import get_ltor_masks_and_position_ids, model_provider

    args = get_args()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    device = torch.device("cuda", torch.cuda.current_device())

    model = model_provider(pre_process=pp_rank == 0, post_process=pp_rank == pp_size - 1)
    model = model.to(device)
    model.train()
    # Pai-Megatron-Patch's Qwen2VLModel never sets this; the schedule dispatcher needs it.
    from megatron.core.enums import ModelType
    model.model_type = ModelType.encoder_or_decoder
    # finalize_model_grads() expects a DDP wrapper. DP=1 here, so these are no-ops.
    from contextlib import nullcontext
    model.finish_grad_sync = lambda: None
    model.start_grad_sync = lambda: None
    model.no_sync = lambda: nullcontext()
    from megatron.core.distributed import DistributedDataParallelConfig
    model.ddp_config = DistributedDataParallelConfig()
    for param in model.parameters():
        if param.requires_grad and getattr(param, "main_grad", None) is None:
            param.main_grad = torch.zeros_like(param, dtype=torch.float32)

    tokens, labels, grid, pixels, image_mask, sample_id = build_batch(
        hf_dir, os.environ["MANIFEST"], repo_root, device)

    target_mc = labels.clone()
    target_mc[:, :-1] = labels[:, 1:]
    target_mc[:, -1] = -100
    empty_video = torch.zeros((0, 3), dtype=torch.long, device=device)
    _, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens, grid, empty_video, target_mc, 0, ignore_index=-100)

    # --- bubble instrumentation -------------------------------------------------
    # busy_events holds one (start, end) CUDA-event pair per forward and per backward
    # call. Everything runs on the default stream, so the intervals are serial on a rank
    # and can simply be summed. See docs/pipeline-bubble.md.
    busy_events = []
    bwd_pre_events = []
    backward_timed = True

    def _fwd_busy_call(fn):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        busy_events.append((start, end))
        return result

    def _bwd_pre_hook(module, grad_output):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        bwd_pre_events.append(ev)

    def _bwd_hook(module, grad_input, grad_output):
        if bwd_pre_events:
            start = bwd_pre_events.pop(0)
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            busy_events.append((start, end))

    if bench_steps > 0:
        # Full backward hooks make the module output a view, which trips MCore's
        # deallocate_output_tensor assert. The pseudo-free is a memory-only optimisation.
        model.config.deallocate_pipeline_outputs = False
        try:
            model.register_full_backward_pre_hook(_bwd_pre_hook)
            model.register_full_backward_hook(_bwd_hook)
        except Exception as exc:  # measurement degrades, the run continues
            backward_timed = False
            print(f"WARN backward timing unavailable: {exc}", flush=True)

    def forward_step(data_iterator, model):
        def _run():
            return model(
                input_ids=tokens,
                position_ids=position_ids,
                vision_data=pixels,
                vision_grid_thw=grid,
                video_start_index=int(image_mask.sum().item()),
                image_input_mask=image_mask,
                video_input_mask=None,
                attention_mask=None,
                labels=target_mc,
            )

        losses = _fwd_busy_call(_run) if bench_steps > 0 else _run()

        def loss_func(output_tensor):
            mask = loss_mask.view(-1).float()
            loss = (output_tensor.float().view(-1) * mask).sum() / mask.sum()
            return loss, {"loss": loss.detach()}

        return losses, loss_func

    def _endless():
        while True:
            yield {}  # forward_step ignores this; data is already on device

    data_iterator = _endless()

    # The schedule ends with finalize_model_grads(), which expects a DDP wrapper
    # (finish_grad_sync). We run unwrapped with DP=1, so grad finalisation is a no-op here.
    import megatron.core.pipeline_parallel.schedules as _sched
    if hasattr(_sched, "finalize_model_grads"):
        _sched.finalize_model_grads = lambda *a, **k: None
    fwd_bwd = get_forward_backward_func()

    def _loss_of(item):
        if isinstance(item, dict):
            return item.get("loss")
        if isinstance(item, (tuple, list)) and item:
            return item[0]
        return item

    steps = warmup_steps + bench_steps if bench_steps > 0 else 1
    records = []
    last_local_loss = float("nan")

    for step_idx in range(steps):
        if bench_steps > 0:
            torch.distributed.barrier()
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
        busy_events.clear()
        bwd_pre_events.clear()
        for param in model.parameters():
            grad = getattr(param, "main_grad", None)
            if grad is not None:
                grad.zero_()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = fwd_bwd(
                forward_step_func=forward_step,
                data_iterator=data_iterator,
                model=[model],
                num_microbatches=num_microbatches,
                seq_length=tokens.shape[1],
                micro_batch_size=1,
                forward_only=False,
            )

        values = [v for v in (_loss_of(x) for x in (out or [])) if v is not None]
        last_local_loss = (float(sum(float(v) for v in values) / len(values))
                           if values else float("nan"))
        # Only the last pipeline stage produces losses; broadcast so every rank agrees.
        loss_t = torch.tensor([last_local_loss], device=device)
        torch.distributed.broadcast(loss_t, src=pp_size - 1)
        step_loss = float(loss_t.item())

        if bench_steps > 0:
            end_ev.record()
            torch.distributed.barrier()
            torch.cuda.synchronize()
            t_local = start_ev.elapsed_time(end_ev) / 1000.0
            busy_local = sum(s.elapsed_time(e) for s, e in busy_events) / 1000.0

            t_t = torch.tensor([t_local], device=device)
            torch.distributed.all_reduce(t_t, op=torch.distributed.ReduceOp.MAX)
            busy_t = torch.tensor([busy_local], device=device)
            torch.distributed.all_reduce(busy_t, op=torch.distributed.ReduceOp.SUM)
            step_time = float(t_t.item())
            busy_total = float(busy_t.item())
            capacity = pp_size * step_time
            bubble_time = max(0.0, capacity - busy_total)
            bubble_ratio = bubble_time / capacity if capacity > 0 else float("nan")

            peak_local = torch.tensor([torch.cuda.max_memory_allocated()],
                                      dtype=torch.float64, device=device)
            peak_all = [torch.zeros_like(peak_local) for _ in range(pp_size)]
            torch.distributed.all_gather(peak_all, peak_local)
            peak_by_rank = {str(r): int(peak_all[r].item()) for r in range(pp_size)}

            if step_idx + 1 == warmup_steps:
                torch.cuda.reset_peak_memory_stats()

            if step_idx >= warmup_steps:
                records.append({
                    "step": step_idx - warmup_steps,
                    "warmup": False,
                    "step_time_s": step_time,
                    "samples": num_microbatches,
                    "max_memory_allocated_bytes_by_rank": peak_by_rank,
                    "busy_time_s_total": busy_total,
                    "bubble_time_s": bubble_time,
                    "pipeline_bubble_ratio": bubble_ratio,
                    "backward_busy_timed": backward_timed,
                    "loss": step_loss,
                    "skipped": not math.isfinite(step_loss),
                    "nan_or_inf": not math.isfinite(step_loss),
                    "pp_size": pp_size,
                    "num_layers": int(num_layers),
                    "num_microbatches": num_microbatches,
                    "mode": "pp_executor" if mode in ("1", "true", "yes", "on")
                            else "native",
                })

    total = torch.zeros(1, device=device)
    for param in model.parameters():
        grad = getattr(param, "main_grad", None)
        if grad is not None:
            total += grad.detach().float().pow(2).sum()
    torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
    grad_norm = math.sqrt(float(total))

    result = {
        "mode": "pp_executor" if mode in ("1", "true", "yes", "on") else "native",
        "use_pp_executor_env": mode,
        "pp_rank": pp_rank,
        "pp_size": pp_size,
        "pp_size_env": pp_size_env,
        "num_layers": num_layers,
        "sample_id": sample_id,
        "num_microbatches": num_microbatches,
        "seq_length": int(tokens.shape[1]),
        "loss": step_loss,
        "grad_norm": grad_norm,
        "finite": bool(math.isfinite(step_loss) and math.isfinite(grad_norm)),
        "measured": [f"PP={pp_size} forward+backward loss and grad norm"],
        "unmeasured": ["encoder nesting (VE executor)", "training convergence"],
    }
    if bench_steps > 0:
        result["benchmark"] = {
            "warmup_steps": warmup_steps,
            "measured_steps": len(records),
            "backward_busy_timed": backward_timed,
            "jsonl": str(out_path.with_suffix(".jsonl")),
        }
        result["measured"].append(
            "step_time_s, throughput, peak allocated GPU bytes, exact pipeline bubble")
    if pp_rank == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if records:
            jsonl = out_path.with_suffix(".jsonl")
            jsonl.write_text("".join(json.dumps(r) + "\n" for r in records))
        elif out_path.with_suffix(".jsonl").exists():
            out_path.with_suffix(".jsonl").unlink()  # no stale metrics beside a gate result
        out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"rank": pp_rank, "mode": result["mode"],
                      "loss": step_loss, "grad_norm": grad_norm,
                      "steps": len(records)}), flush=True)


if __name__ == "__main__":
    main()

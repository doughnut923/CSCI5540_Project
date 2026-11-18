"""Run the same real-image Qwen2-VL forward/backward check on MPS, CPU or CUDA."""

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import signal
import sys
import threading
import time
import traceback

from prepare import verify_reference


ROOT = Path(__file__).resolve().parent


class Tee:
    def __init__(self, terminal, log):
        self.terminal = terminal
        self.log = log

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class MemoryMonitor:
    """Sample MPS/process memory. CUDA also exposes an exact allocator peak."""

    def __init__(self, torch, device):
        import psutil

        self.torch = torch
        self.device = device
        self.process = psutil.Process()
        self.stop_event = threading.Event()
        self.peak = {}
        self.thread = threading.Thread(target=self.sample_loop, daemon=True)

    def snapshot(self):
        values = {"process_rss_bytes": self.process.memory_info().rss}
        if self.device.type == "mps":
            values["mps_allocated_bytes"] = self.torch.mps.current_allocated_memory()
            values["mps_driver_bytes"] = self.torch.mps.driver_allocated_memory()
        elif self.device.type == "cuda":
            values["cuda_allocated_bytes"] = self.torch.cuda.memory_allocated(self.device)
            values["cuda_reserved_bytes"] = self.torch.cuda.memory_reserved(self.device)
            values["cuda_allocator_peak_bytes"] = self.torch.cuda.max_memory_allocated(self.device)
        return values

    def sample_loop(self):
        while not self.stop_event.is_set():
            for name, value in self.snapshot().items():
                self.peak[name] = max(value, self.peak.get(name, 0))
            self.stop_event.wait(0.1)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        return {"sampling_interval_seconds": 0.1, "sampled_peak": self.peak,
                "final": self.snapshot(), "mps_peak_method": "sampled, not an exact allocator peak"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16"), default="auto")
    parser.add_argument("--mode", choices=("train-step", "backward"), default="train-step")
    parser.add_argument("--cases", help="Comma-separated case names from config.json")
    parser.add_argument("--steps", type=int, help="Steps per case")
    parser.add_argument("--visual-token-budget", type=int)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--mps-fallback", action="store_true", help="Explicitly allow PyTorch's CPU fallback")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def synchronize(torch, device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def select_device(torch, requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def environment(torch, device):
    import psutil

    packages = {}
    for name in ("torch", "torchvision", "transformers", "numpy", "Pillow", "accelerate", "safetensors", "huggingface-hub", "tokenizers", "jinja2"):
        packages[name] = importlib.metadata.version(name)
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "device": str(device), "packages": packages,
            "system_memory_bytes": psutil.virtual_memory().total,
            "initial_available_memory_bytes": psutil.virtual_memory().available,
            "mps_fallback_environment": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0")}
    if device.type == "cuda":
        info.update(cuda_version=torch.version.cuda, gpu_name=torch.cuda.get_device_name(device),
                    gpu_memory_bytes=torch.cuda.get_device_properties(device).total_memory)
    elif device.type == "mps":
        info["mps_recommended_max_memory_bytes"] = torch.mps.recommended_max_memory()
    return info


def prepare_inputs(processor, case, config, torch):
    from PIL import Image

    manifest = {item["file"]: item for item in json.loads((ROOT / "fixtures/photo_manifest.json").read_text())}
    images, image_info = [], []
    for relative in case["photos"]:
        path = ROOT / "fixtures" / relative
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != manifest[relative]["sha256"]:
            raise ValueError(f"Fixture hash differs: {relative}")
        with Image.open(path) as source:
            image = source.convert("RGB")
        images.append(image)
        image_info.append({"file": relative, "sha256": actual_hash, "source_wh": list(image.size)})
    count = len(images)
    processor.image_processor.min_pixels = config["min_pixels"]
    processor.image_processor.max_pixels = (config["visual_token_budget"] // count) * 28 * 28
    if processor.image_processor.max_pixels < config["min_pixels"]:
        raise ValueError("Visual-token budget is too small for this image count")
    message = {"role": "user", "content": [{"type": "image"} for _ in images] +
               [{"type": "text", "text": "Briefly describe the image content."}]}
    prefix = processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
    full = processor.apply_chat_template([message, {"role": "assistant", "content":
        [{"type": "text", "text": case["answer"]}]}], tokenize=False, add_generation_prompt=False)
    batch = dict(processor(text=[full], images=images, padding=True, return_tensors="pt"))
    prefix_batch = processor(text=[prefix], images=images, padding=True, return_tensors="pt")
    prefix_length = prefix_batch["input_ids"].shape[1]
    if not torch.equal(batch["input_ids"][:, :prefix_length], prefix_batch["input_ids"]):
        raise ValueError("Assistant loss-mask boundary does not match the full prompt")
    labels = batch["input_ids"].clone()
    labels[:, :prefix_length] = -100
    labels[batch["attention_mask"] == 0] = -100
    supervised = int((labels != -100).sum())
    if supervised < 2:
        raise ValueError("The example must contain supervised assistant tokens")
    batch["labels"] = labels
    grid = batch["image_grid_thw"]
    merge = processor.image_processor.merge_size
    per_image = (grid.prod(dim=1) // (merge * merge)).tolist()
    image_token = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    placeholders = int((batch["input_ids"] == image_token).sum())
    if len(grid) != count or placeholders != sum(per_image):
        raise ValueError("Image count, grids and image placeholders disagree")
    info = {"photos": image_info, "image_count": count, "image_grid_thw": grid.tolist(),
            "derived_visual_tokens_per_image": per_image,
            "derived_visual_tokens_total": sum(per_image), "image_placeholder_count": placeholders,
            "supervised_text_tokens": supervised, "prompt_tokens": prefix_length,
            "tensor_shapes": {k: list(v.shape) for k, v in batch.items()},
            "processor_min_pixels": processor.image_processor.min_pixels,
            "processor_max_pixels": processor.image_processor.max_pixels}
    return batch, info


def parameter_group(name):
    if name.startswith("visual.merger."):
        return "vision_merger"
    if name.startswith("visual."):
        return "vision_encoder"
    return "language_model"


def inspect_gradients(model, torch):
    stats = {name: {"parameters": 0, "missing_gradients": [], "norms": [], "finite": [], "nonzero": []}
             for name in ("vision_encoder", "vision_merger", "language_model")}
    selected = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            group = parameter_group(name)
            entry = stats[group]
            entry["parameters"] += parameter.numel()
            if parameter.grad is None:
                entry["missing_gradients"].append(name)
                continue
            gradient = parameter.grad.detach()
            entry["finite"].append(torch.isfinite(gradient).all())
            entry["nonzero"].append(torch.count_nonzero(gradient) > 0)
            entry["norms"].append(torch.linalg.vector_norm(gradient, dtype=torch.float32))
            # Use matrix weights to observe updates. Avoid copying full parameters.
            if group not in selected and parameter.ndim >= 2 and not name.startswith("model.embed_tokens."):
                indices = torch.linspace(0, parameter.numel() - 1, steps=min(4096, parameter.numel()),
                                         device=parameter.device).long()
                selected[group] = (name, parameter, indices, parameter.detach().flatten()[indices].float().cpu().clone())
    result = {}
    for group, entry in stats.items():
        finite = bool(torch.stack(entry["finite"]).all().item()) if entry["finite"] else False
        nonzero = bool(torch.stack(entry["nonzero"]).any().item()) if entry["nonzero"] else False
        norm = float(torch.linalg.vector_norm(torch.stack(entry["norms"])).item()) if entry["norms"] else 0.0
        result[group] = {"trainable_parameters": entry["parameters"], "all_gradients_finite": finite,
                         "any_gradient_nonzero": nonzero, "gradient_l2": norm,
                         "missing_gradient_names": entry["missing_gradients"]}
        if not finite or not nonzero or entry["missing_gradients"] or not math.isfinite(norm):
            raise AssertionError(f"Gradient check failed for {group}: {result[group]}")
    return result, selected


def inspect_updates(selected):
    updates = {}
    for group, (name, parameter, indices, before) in selected.items():
        after = parameter.detach().flatten()[indices].float().cpu()
        if not bool(after.isfinite().all()):
            raise AssertionError(f"Updated parameter is non-finite: {name}")
        difference = (after - before).abs()
        updates[group] = {"parameter": name, "sampled_elements": before.numel(),
                          "changed_elements": int((difference > 0).sum()),
                          "max_abs_change": float(difference.max())}
        if not updates[group]["changed_elements"]:
            raise AssertionError(f"No sampled parameter update in {group}: {name}")
    if len(updates) != 3:
        raise AssertionError("Expected parameter-update evidence from all three module groups")
    return updates


def run(args, config, result):
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from mps_compat import FlattenMPSScatterSource

    device = select_device(torch, args.device)
    dtype_name = args.dtype
    if dtype_name == "auto":
        supports_bfloat16 = device.type == "mps" or (device.type == "cuda" and torch.cuda.is_bf16_supported())
        dtype_name = "bfloat16" if supports_bfloat16 else "float32"
    dtype = getattr(torch, dtype_name)
    result["environment"] = environment(torch, device)
    result["dtype"] = dtype_name
    result["mode"] = args.mode
    result["compatibility"] = {"mps_scatter_source_flat_view": device.type == "mps"}
    result["config"] = config
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    # Verify the selected precision before loading the multi-GB model.
    probe = torch.ones((2, 2), device=device, dtype=dtype, requires_grad=True)
    probe.square().mean().backward()
    del probe
    if args.model_dir:
        model_dir = args.model_dir.resolve()
    else:
        prepared = ROOT / ".cache/prepared.json"
        if not prepared.exists():
            raise FileNotFoundError("Run python prepare.py first, or pass --model-dir")
        model_dir = Path(json.loads(prepared.read_text())["model_dir"])
    reference = json.loads((ROOT / "fixtures/reference_assets.json").read_text())
    result["checkpoint_files"] = verify_reference(model_dir, reference)
    monitor = MemoryMonitor(torch, device)
    monitor.start()
    handle = None
    try:
        print(f"Loading {config['model_id']} on {device} with {dtype_name}", flush=True)
        started = time.perf_counter()
        model, loading_info = Qwen2VLForConditionalGeneration.from_pretrained(
            model_dir, local_files_only=True, torch_dtype=dtype,
            attn_implementation=config["attention_implementation"], low_cpu_mem_usage=True,
            output_loading_info=True,
        )
        loading_info = {key: sorted(value) if isinstance(value, set) else value
                        for key, value in loading_info.items()}
        result["checkpoint_loading"] = loading_info
        if any(loading_info.values()):
            raise ValueError(f"Checkpoint loading was incomplete: {loading_info}")
        model = model.to(device)
        model.train()
        model.requires_grad_(True)
        model.config.use_cache = False
        if config["gradient_checkpointing"]:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        synchronize(torch, device)
        result["load_seconds"] = time.perf_counter() - started
        result["model_parameters"] = sum(p.numel() for p in model.parameters())
        result["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        result["checkpointing"] = {"vision": model.visual.gradient_checkpointing,
                                   "language": model.model.gradient_checkpointing,
                                   "use_reentrant": False}
        result["attention_classes"] = {
            "vision": type(model.visual.blocks[0].attn).__name__,
            "language": type(model.model.layers[0].self_attn).__name__,
        }
        print(f"Loaded {result['model_parameters']:,} trainable parameters", flush=True)
        captured = {}

        def capture_visual_output(module, inputs, output):
            # Capture the stage boundary immediately. PyTorch 2.6 MPS
            # masked_scatter may later flatten the source tensor metadata.
            captured["output_shape"] = list(output.shape)
            output.retain_grad()
            captured["output"] = output

        handle = model.visual.register_forward_hook(capture_visual_output)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
            betas=tuple(config["betas"]), eps=config["epsilon"], weight_decay=config["weight_decay"],
            foreach=False, fused=False)
        processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
        result["cases"] = []
        for case in config["cases"]:
            batch, info = prepare_inputs(processor, case, config, torch)
            batch = {name: tensor.to(device=device, dtype=dtype if tensor.is_floating_point() else tensor.dtype)
                     for name, tensor in batch.items()}
            case_result = {"name": case["name"], "input": info, "steps": []}
            result["cases"].append(case_result)
            print(f"Case {case['name']}: {info['image_count']} images, {info['derived_visual_tokens_total']} visual tokens", flush=True)
            for step in range(config["steps"]):
                optimizer.zero_grad(set_to_none=True)
                captured.clear()
                synchronize(torch, device)
                started = time.perf_counter()
                scatter_fix = FlattenMPSScatterSource() if device.type == "mps" else None
                with scatter_fix if scatter_fix is not None else contextlib.nullcontext():
                    output = model(**batch, use_cache=False, return_dict=True)
                synchronize(torch, device)
                forward_seconds = time.perf_counter() - started
                loss = float(output.loss.detach().float().cpu())
                if not math.isfinite(loss):
                    raise AssertionError(f"Non-finite loss: {loss}")
                vision = captured["output"]
                case_result["forward_observation"] = {
                    "loss": loss,
                    "vision_shape_at_encoder_boundary": captured["output_shape"],
                    "vision_shape_after_llm_input_scatter": list(vision.shape),
                    "expected_visual_tokens": info["derived_visual_tokens_total"],
                }
                boundary_shape = captured["output_shape"]
                print(f"  visual output {boundary_shape}, expected {info['derived_visual_tokens_total']} tokens", flush=True)
                if boundary_shape != [info["derived_visual_tokens_total"], model.config.hidden_size]:
                    raise AssertionError(f"Actual vision shape {boundary_shape} differs from expected visual tokens and hidden size")
                print(f"  step {step + 1}: loss={loss:.6f}; backward", flush=True)
                started = time.perf_counter()
                output.loss.backward()
                synchronize(torch, device)
                backward_seconds = time.perf_counter() - started
                gradients, selected = inspect_gradients(model, torch)
                if vision.grad is None or not bool(torch.isfinite(vision.grad).all().item()):
                    raise AssertionError("Gradient did not reach the visual output")
                vision_grad_norm = float(torch.linalg.vector_norm(vision.grad, dtype=torch.float32).item())
                if not math.isfinite(vision_grad_norm) or vision_grad_norm <= 0:
                    raise AssertionError("Visual-output gradient is zero or non-finite")
                record = {"step": step + 1, "loss": loss, "forward_seconds": forward_seconds,
                          "backward_seconds": backward_seconds, "gradients": gradients,
                          "vision_output_shape": boundary_shape,
                          "vision_output_gradient_shape": list(vision.grad.shape),
                          "vision_output_gradient_l2": vision_grad_norm,
                          "mps_scatter_fix_calls": scatter_fix.applied_calls if scatter_fix else 0}
                if args.mode == "train-step":
                    print("  gradients valid; AdamW step", flush=True)
                    started = time.perf_counter()
                    optimizer.step()
                    synchronize(torch, device)
                    record["optimizer_seconds"] = time.perf_counter() - started
                    state = next(iter(optimizer.state.values()))
                    record["optimizer_state_dtypes"] = {name: str(value.dtype)
                        for name, value in state.items() if torch.is_tensor(value)}
                    record["updates"] = inspect_updates(selected)
                record["memory"] = monitor.snapshot()
                case_result["steps"].append(record)
                print(f"  step {step + 1}: passed", flush=True)
                del output, vision, selected
                captured.clear()
            optimizer.zero_grad(set_to_none=True)
            del batch
        result["status"] = "passed"
    finally:
        if handle is not None:
            handle.remove()
        result["memory"] = monitor.stop()


def main():
    args = parse_args()
    if args.mps_fallback:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    config = json.loads(args.config.read_text())
    reference = json.loads((ROOT / "fixtures/reference_assets.json").read_text())
    if (config["model_id"], config["revision"]) != (reference["repo"], reference["revision"]):
        raise ValueError("The model and processor must match the pinned shared reference")
    if config["optimizer"] != "adamw" or config["trainable_scope"] != "all_parameters":
        raise ValueError("This checker supports AdamW and all_parameters. Keep configuration and execution consistent.")
    for argument, key in ((args.steps, "steps"), (args.visual_token_budget, "visual_token_budget"),
                          (args.gradient_checkpointing, "gradient_checkpointing")):
        if argument is not None:
            config[key] = argument
    if args.cases:
        wanted = args.cases.split(",")
        available = {case["name"] for case in config["cases"]}
        if not set(wanted) <= available:
            raise ValueError(f"Unknown cases: {set(wanted) - available}")
        config["cases"] = [case for case in config["cases"] if case["name"] in wanted]
    if config["steps"] < 1 or config["visual_token_budget"] < 1:
        raise ValueError("Steps and visual-token budget must be positive")
    output_dir = args.output_dir or ROOT / "runs" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    result = {"status": "running", "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "command_arguments": vars(args).copy(), "scope": "Single-device correctness check for the complete model",
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ("run_check.py", "mps_compat.py", "prepare.py", "config.json", "requirements.txt")}}
    result["config_source_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    def save_result():
        temporary = output_dir / "result.json.tmp"
        temporary.write_text(json.dumps(result, indent=2, default=str, allow_nan=False) + "\n")
        temporary.replace(output_dir / "result.json")

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    save_result()
    exit_code = 0
    with (output_dir / "run.log").open("w", buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                run(args, config, result)
            except KeyboardInterrupt as error:
                result["status"] = "interrupted"
                result["error"] = {"type": type(error).__name__, "message": str(error)}
                exit_code = 130
            except Exception as error:
                result["status"] = "failed"
                result["error"] = {"type": type(error).__name__, "message": str(error)}
                traceback.print_exc()
                exit_code = 1
            finally:
                result["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
                save_result()
                print(f"Result: {result['status']} ({output_dir / 'result.json'})", flush=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

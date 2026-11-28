"""Run the upstream BigMac multimodal executor with two Qwen2-VL pipeline stages."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import distributed as dist, nn
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.transformer.transformer_config import TransformerConfig
from pp_executor.encoder_output_meta import EncoderOutputMeta
from pp_executor.llm_pp_executor import LLMPPExecutorWithVE
from transformers import AutoModelForImageTextToText, AutoTokenizer, Qwen2VLImageProcessor

from native_qwen2vl_smoke import assemble

SEQ = 640
VISUAL = 576
HIDDEN = 1536
DTYPE = torch.bfloat16
# The two PP ranks each encode one real microbatch per step.
PAIRS = ((0, 1), (1, 2), (2, 0))


def embeddings_hashes(embeddings, lengths):
    return [
        hashlib.sha256(part.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        for part in embeddings.split(lengths)
    ]


def embedding_grad_norms(grad, lengths):
    return [part.float().norm().item() for part in grad.split(lengths)]


class TextStage(nn.Module):
    def __init__(self, text, head, rank):
        super().__init__()
        self.rank = rank
        self.text = text
        self.head = head if rank == 1 else None
        self.pre_process = rank == 0
        self.post_process = rank == 1
        self.share_embeddings_and_output_weights = False
        self.model_type = ModelType.encoder_or_decoder
        self.ddp_config = DistributedDataParallelConfig()
        self.config = TransformerConfig(
            num_layers=len(text.layers), hidden_size=HIDDEN, num_attention_heads=12,
            bf16=True, params_dtype=DTYPE, pipeline_dtype=DTYPE,
            pipeline_model_parallel_size=2, deallocate_pipeline_outputs=False,
        )
        midpoint = len(text.layers) // 2
        stage_slice = slice(0, midpoint) if rank == 0 else slice(midpoint, None)
        text.layers = nn.ModuleList(list(text.layers)[stage_slice])
        text.config.layer_types = text.config.layer_types[stage_slice]
        if rank == 0:
            text.norm = nn.Identity()

    def set_input_tensor(self, value):
        self.pipeline_input = value[0] if isinstance(value, list) else value

    def finish_grad_sync(self, force_all_reduce=False):
        # Text layers are frozen and DP size is one. Vision replicas sync explicitly.
        pass

    def forward(self, inputs, sample, evidence):
        if self.pre_process:
            visual = inputs["vision_embeds"]
            mask = inputs["image_mask"]
            assert mask.sum().item() == VISUAL
            visual.register_hook(
                lambda grad: evidence.update(llm_visual_grad_norm=embedding_grad_norms(grad, sample["tokens_per_image"]))
            )
            base = self.text.embed_tokens(inputs["ids"])
            hidden = base.masked_scatter(mask.unsqueeze(-1), visual.to(base.dtype))
            evidence["llm_embedding_sha256"] = embeddings_hashes(hidden[mask], sample["tokens_per_image"])
        else:
            assert self.pipeline_input is not None
            hidden = self.pipeline_input.transpose(0, 1).contiguous()
            evidence["received_stage_activation"] = list(self.pipeline_input.shape)

        output = self.text(
            inputs_embeds=hidden, attention_mask=inputs["attention"],
            position_ids=inputs["positions"], use_cache=False,
        ).last_hidden_state
        if self.pre_process:
            return output.transpose(0, 1).contiguous()

        labels = inputs["labels"][:, 1:]
        supervised = labels != -100
        logits = self.head(output[:, :-1][supervised])
        loss = F.cross_entropy(logits.float(), labels[supervised])
        assert torch.isfinite(loss).item()
        evidence["loss"] = loss.detach().item()
        return loss


class QwenBridge:
    def __init__(self, model, stage, samples, tokenizer, processor, rank):
        self.model = model
        self.stage = stage
        self.rank = rank
        self.samples = samples
        self.prepared = []
        for sample in samples:
            batch, mask = assemble(sample, processor, tokenizer)
            padding = SEQ - batch["input_ids"].shape[1]
            assert padding >= 0
            ids = F.pad(batch["input_ids"], (0, padding), value=tokenizer.pad_token_id)
            attention = F.pad(batch["attention_mask"], (0, padding))
            labels = F.pad(batch["labels"], (0, padding), value=-100)
            image_mask = F.pad(mask, (0, padding))
            grids = batch["image_grid_thw"]
            positions, _ = model.model.get_rope_index(
                input_ids=ids, mm_token_type_ids=image_mask.long(),
                attention_mask=attention, image_grid_thw=grids,
            )
            assert positions.shape == (3, 1, SEQ)
            text_positions = torch.arange(SEQ).view(1, 1, SEQ)
            positions = torch.cat((text_positions, positions), dim=0)
            self.prepared.append((batch["pixel_values"], grids, ids, attention, labels, image_mask, positions))
        self.vision = {}
        self.language = {}
        self.next_id = 0

    def metas(self):
        return [
            EncoderOutputMeta("vision_embeds", requires_grad=True, shape=(VISUAL, HIDDEN),
                              dtype=DTYPE, only_requires_in_first_stage=True),
            EncoderOutputMeta("ids", shape=(1, SEQ), dtype=torch.long),
            EncoderOutputMeta("attention", shape=(1, SEQ), dtype=torch.long),
            EncoderOutputMeta("labels", shape=(1, SEQ), dtype=torch.long),
            EncoderOutputMeta("image_mask", shape=(1, SEQ), dtype=torch.bool),
            EncoderOutputMeta("positions", shape=(4, 1, SEQ), dtype=torch.long),
        ]

    def encoder_forward(self, encoder_models, _data, microbatch_id):
        assert microbatch_id % 2 == self.rank
        pixels, grids, *metadata = [value.cuda().contiguous() for value in self.prepared[microbatch_id]]
        with torch.autocast("cuda", dtype=DTYPE):
            encoded = encoder_models[0].module(pixels.to(DTYPE), grid_thw=grids).pooler_output
        encoded = encoded.to(DTYPE).contiguous()
        assert encoded.requires_grad and encoded.shape == (VISUAL, HIDDEN)
        sample = self.samples[microbatch_id]
        record = {
            "encoder_rank": self.rank, "grid_thw": sample["grid_thw"],
            "tokens_per_image": sample["tokens_per_image"],
            "encoder_embedding_sha256": embeddings_hashes(encoded, sample["tokens_per_image"]),
        }
        encoded.register_hook(
            lambda grad: record.update(encoder_grad_norm=embedding_grad_norms(grad, sample["tokens_per_image"]))
        )
        self.vision[microbatch_id] = record
        return [encoded, *metadata]

    def forward_step(self, data_iterator, stage):
        microbatch_id = self.next_id
        self.next_id += 1
        self.language[microbatch_id] = {}
        with torch.autocast("cuda", dtype=DTYPE):
            value = stage(next(data_iterator), self.samples[microbatch_id], self.language[microbatch_id])
        return value, lambda loss: (loss, {"loss": loss.detach().clone()})


def merged_evidence(bridges, samples):
    output = []
    for i, sample in enumerate(samples):
        source = bridges[i % 2]["vision"][i]
        first = bridges[0]["language"][i]
        last = bridges[1]["language"][i]
        assert source["encoder_embedding_sha256"] == first["llm_embedding_sha256"]
        count = sample["image_count"]
        assert len(set(source["encoder_embedding_sha256"])) == count
        for norms in (source["encoder_grad_norm"], first["llm_visual_grad_norm"]):
            assert len(norms) == count and all(torch.isfinite(torch.tensor(v)).item() and v > 0 for v in norms)
        assert last["received_stage_activation"] == [SEQ, 1, HIDDEN]
        output.append({
            "sample_id": sample["sample_id"], "source_rows": [item["source_row"] for item in sample["images"]],
            "llm_visual_tokens": VISUAL, "encoder_llm_embeddings_match": True, **source, **first, **last,
        })
    return output


def run(args):
    rank = int(os.environ["LOCAL_RANK"])
    assert torch.cuda.device_count() == 2
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    torch.manual_seed(55402026)
    dist.init_process_group("nccl", timeout=timedelta(seconds=85), device_id=torch.device("cuda", rank))
    try:
        parallel_state.initialize_model_parallel(pipeline_model_parallel_size=2)
        group = parallel_state.get_pipeline_model_parallel_group()
        assert dist.get_world_size(group) == 2 and parallel_state.get_pipeline_model_parallel_rank() == rank
        manifest = json.loads(args.manifest.read_text())
        assert args.model_dir.name == manifest["model_revision"]
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_dir, local_files_only=True, dtype=DTYPE, device_map={"": f"cuda:{rank}"},
            attn_implementation="sdpa",
        )
        model.requires_grad_(False)
        model.train()
        parameter = model.model.visual.merger.mlp[-1].weight
        parameter.data = parameter.detach().float()
        parameter.requires_grad_(True)
        original = parameter.detach().clone()
        stage = TextStage(model.model.language_model, model.lm_head, rank)
        print(json.dumps({
            "stage_ready": rank, "hostname": os.uname().nodename, "device": torch.cuda.get_device_name(rank),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "decoder_layers": len(stage.text.layers),
            "executor_source": __import__("inspect").getfile(LLMPPExecutorWithVE),
            "checkpoint_revision": manifest["model_revision"],
        }), flush=True)
        for count in (1, 4):
            with torch.no_grad():
                parameter.copy_(original)
            optimizer = torch.optim.AdamW([parameter], lr=1e-4, weight_decay=0)
            processor = Qwen2VLImageProcessor.from_pretrained(
                args.model_dir, local_files_only=True, min_pixels=3136, max_pixels=VISUAL // count * 784,
            )
            samples = [item for item in manifest["samples"] if item["image_count"] == count]
            for step, pair in enumerate(PAIRS[:args.steps], 1):
                selected = [samples[index] for index in pair]
                bridge = QwenBridge(model, stage, selected, tokenizer, processor, rank)
                executor = LLMPPExecutorWithVE(
                    forward_step_func=bridge.forward_step, data_iterator=iter(()), model=stage,
                    num_microbatches=2, seq_length=SEQ, micro_batch_size=1, forward_only=False,
                    pp_size=2, pp_group=group, pp_rank=rank,
                    encoder_models=[SimpleNamespace(module=model.model.visual)],
                    encoder_forward_func=bridge.encoder_forward, get_encoder_output_meta_func=bridge.metas,
                    trace_enabled=True, trace_dir=str(args.trace_dir), trace_iter_id=count * 10 + step,
                )
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats()
                before = parameter.detach().clone()
                executor.execute()
                assert len(bridge.vision) == 1 and len(bridge.language) == 2
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
                local_norm = parameter.grad.norm().item()
                assert local_norm > 0
                dist.all_reduce(parameter.grad, group=group)
                optimizer.step()
                delta = parameter.detach() - before
                update = delta.norm().item()
                assert torch.isfinite(delta).all().item() and update > 0
                evidence = [None, None]
                dist.all_gather_object(evidence, {
                    "vision": bridge.vision, "language": bridge.language, "rank": rank,
                    "local_projector_grad_l2": local_norm, "projector_update_l2": update,
                    "parameter_sha256": hashlib.sha256(parameter.detach().cpu().numpy().tobytes()).hexdigest(),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                }, group=group)
                if rank == 0:
                    assert evidence[0]["parameter_sha256"] == evidence[1]["parameter_sha256"]
                    record = {
                        "status": "PASS", "framework": "upstream LLMPPExecutorWithVE", "image_count": count,
                        "optimizer_step": step, "microbatches": merged_evidence(evidence, selected),
                        "ranks": [{key: val for key, val in entry.items() if key not in ("vision", "language")}
                                  for entry in evidence],
                    }
                    with args.output.open("a") as output:
                        output.write(json.dumps(record) + "\n")
                    print(json.dumps(record), flush=True)
                dist.barrier(group=group)
                stage.pipeline_input = None
                del executor, bridge
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=(2, 3), default=2)
    args = parser.parse_args()
    assert not args.output.exists()
    run(args)

"""Time paired Part C workloads using the original two-rank BigMac smoke adapter."""
from datetime import timedelta
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import distributed as dist
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = Path(os.environ['PARTC_PREVIOUS_CODE'])
sys.path.insert(0, str(PREVIOUS))
spec = importlib.util.spec_from_file_location('previous_smoke', PREVIOUS / 'bigmac_qwen2vl.py')
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class TimedStage(smoke.TextStage):
    def forward(self, inputs, sample, evidence):
        if self.pre_process:
            base = self.text.embed_tokens(inputs['ids'])
            hidden = base.masked_scatter(inputs['image_mask'].unsqueeze(-1), inputs['vision_embeds'].to(base.dtype))
        else:
            hidden = self.pipeline_input.transpose(0, 1).contiguous()
        output = self.text(inputs_embeds=hidden, attention_mask=inputs['attention'],
                           position_ids=inputs['positions'], use_cache=False).last_hidden_state
        if self.pre_process:
            return output.transpose(0, 1).contiguous()
        labels = inputs['labels'][:, 1:]
        supervised = labels != -100
        logits = self.head(output[:, :-1][supervised])
        loss = F.cross_entropy(logits.float(), labels[supervised])
        # MCore divides the returned loss in place before backward. Both runtimes
        # record the same raw cross entropy before their common microbatch scaling.
        evidence['loss'] = loss.detach().clone()
        return loss


def load_input(data_dir, key, model):
    with np.load(data_dir / 'tensors' / f'{key}.npz', allow_pickle=False) as raw:
        ids, attention, labels = [torch.from_numpy(raw[name].copy()).long().reshape(1, -1)
                                  for name in ('input_ids', 'attention_mask', 'labels')]
        grid = torch.from_numpy(raw['image_grid_thw'].copy()).long().reshape(-1, 3)
        pixels = torch.from_numpy(raw['pixel_values'].copy()).reshape(-1, 1176)
    mask = ids == model.config.image_token_id
    assert int(mask.sum()) == 576 and int(grid.prod(dim=1).sum()) == 2304
    assert labels.shape == attention.shape == ids.shape and pixels.shape == (2304, 1176)
    assert int(labels.ne(-100).sum()) > 0
    positions, _ = model.model.get_rope_index(input_ids=ids, mm_token_type_ids=mask.long(),
                                               attention_mask=attention, image_grid_thw=grid)
    length = ids.shape[1]
    positions = torch.cat((torch.arange(length).reshape(1, 1, length), positions), dim=0)
    return pixels, grid, ids, attention, labels, mask, positions


class TimedBridge(smoke.QwenBridge):
    def __init__(self, stage, prepared, rank):
        self.stage = stage
        self.prepared = prepared
        self.samples = [None, None]
        self.rank = rank
        self.next_id = 0
        self.language = {}

    def encoder_forward(self, encoder_models, _data, microbatch_id):
        assert microbatch_id % 2 == self.rank
        pixels, grids, *metadata = [v.cuda().contiguous() for v in self.prepared[microbatch_id]]
        with torch.autocast('cuda', dtype=smoke.DTYPE):
            encoded = encoder_models[0].module(pixels.to(smoke.DTYPE), grid_thw=grids).pooler_output
        return [encoded.to(smoke.DTYPE).contiguous(), *metadata]


def execute_step(stage, model, rank, group, parameter, optimizer, selected):
    length = selected[0][2].shape[-1]
    assert selected[1][2].shape[-1] == length
    smoke.SEQ = length
    bridge = TimedBridge(stage, selected, rank)
    executor = smoke.LLMPPExecutorWithVE(
        forward_step_func=bridge.forward_step, data_iterator=iter(()), model=stage,
        num_microbatches=2, seq_length=length, micro_batch_size=1, forward_only=False,
        pp_size=2, pp_group=group, pp_rank=rank,
        encoder_models=[SimpleNamespace(module=model.model.visual)],
        encoder_forward_func=bridge.encoder_forward, get_encoder_output_meta_func=bridge.metas,
        trace_enabled=False,
    )
    dist.barrier(group=group)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    executor.execute()
    dist.all_reduce(parameter.grad, group=group)
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated()
    losses = [bridge.language[i]['loss'].item() for i in range(2)] if rank == 1 else []
    gathered = [None, None]
    dist.all_gather_object(gathered, {'seconds': elapsed, 'peak_bytes': peak, 'losses': losses}, group=group)
    assert all(math.isfinite(value) for value in gathered[1]['losses'])
    stage.pipeline_input = None
    return {'seconds': max(item['seconds'] for item in gathered), 'peak_bytes': [g['peak_bytes'] for g in gathered],
            'losses': gathered[1]['losses'], 'seq_len': length}


def run():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    dist.init_process_group('nccl', timeout=timedelta(seconds=110), device_id=torch.device('cuda', rank))
    smoke.parallel_state.initialize_model_parallel(pipeline_model_parallel_size=2)
    try:
        group = smoke.parallel_state.get_pipeline_model_parallel_group()
        model = smoke.AutoModelForImageTextToText.from_pretrained(os.environ['PARTC_MODEL_DIR'],
                    local_files_only=True, dtype=smoke.DTYPE, device_map={'': f'cuda:{rank}'}, attn_implementation='sdpa')
        model.requires_grad_(False)
        model.train()
        parameter = model.model.visual.merger.mlp[-1].weight
        parameter.data = parameter.detach().float()
        parameter.requires_grad_(True)
        original = parameter.detach().cpu().clone()
        stage = TimedStage(model.model.language_model, model.lm_head, rank)
        plan = json.loads((ROOT / 'data/plan.json').read_text())
        low, medium, high = 'image_count_low', 'image_count_medium', 'image_count_high'
        arms = (low, medium, high)
        # Reversing each pair of orders balances early/late position for all three arms.
        orders = ((low, medium, high), (high, medium, low),
                  (medium, high, low), (low, high, medium))
        batches = {key: load_input(ROOT / 'data', key, model) for key in sorted({
                   s for arm in arms for pair in plan[arm] for s in pair})}
        assert all(len(plan[arm]) == 9 and len({s for pair in plan[arm][1:] for s in pair}) == 12 for arm in arms)
        destination = Path(os.environ['PARTC_RUN_DIR']) / 'primary_results'
        output = destination / 'steps.jsonl'
        if rank == 0:
            output.touch(exist_ok=False)
            (destination / 'workloads.json').write_text(json.dumps(arms) + '\n')
            print('Starting balanced workload blocks, two microbatches per step.', flush=True)
        for block, order in enumerate(orders, 1):
            for arm in order:
                torch.manual_seed(1234)
                with torch.no_grad():
                    parameter.copy_(original)
                optimizer = torch.optim.AdamW([parameter], lr=1e-4, weight_decay=0)
                for i, keys in enumerate(plan[arm]):
                    row = execute_step(stage, model, rank, group, parameter, optimizer, [batches[k] for k in keys])
                    if rank == 0:
                        with output.open('a') as out:
                            out.write(json.dumps({'block':block,'case':arm,'step':i,'warmup':i==0,'ids':keys,**row})+'\n')
                assert torch.isfinite(parameter.grad).all() and parameter.grad.norm().item() > 0
                assert not torch.equal(parameter.detach().cpu(), original)
                del optimizer
                parameter.grad = None
                if rank == 0:
                    print('Complete', block, arm, flush=True)
    finally:
        smoke.parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == '__main__':
    run()

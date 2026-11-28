"""Benchmark paired BigMac schedules on one two-rank allocation."""
import json
import os
import random
from datetime import timedelta
from pathlib import Path

import torch
from torch import distributed as dist
from torch.nn import functional as F
from bigmac_shared import shared, smoke
from plans import ARMS, SEEDS, make_plan
from summarize import write_summary

DATA = Path(os.environ['PARTC_PRIMARY']) / 'data'
OUT = Path(os.environ['PARTC_RUN_DIR']) / 'results'


def pad_batch(values, length, pad_id):
    pixels, grids, ids, attention, labels, mask, positions = values
    old_len = ids.shape[-1]
    assert old_len <= length
    pad = (0, length - old_len)
    text_positions = torch.arange(length).reshape(1, 1, length)
    padded_positions = torch.cat((text_positions, F.pad(positions[1:], pad)), dim=0)
    result = pixels, grids, F.pad(ids,pad,value=pad_id), F.pad(attention,pad), F.pad(labels,pad,value=-100), F.pad(mask,pad), padded_positions
    assert all(torch.equal(a, b[..., :old_len]) for a, b in zip(values[2:], result[2:]))
    assert not result[3][:, old_len:].any() and not result[5][:, old_len:].any()
    assert (result[4][:, old_len:] == -100).all()
    return result


def append(name, value):
    with (OUT/name).open('a') as out:
        out.write(json.dumps(value, allow_nan=False) + '\n')


def run():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    dist.init_process_group('nccl', timeout=timedelta(minutes=7), device_id=torch.device('cuda', rank))
    smoke.parallel_state.initialize_model_parallel(pipeline_model_parallel_size=2)
    try:
        group = smoke.parallel_state.get_pipeline_model_parallel_group()
        model = smoke.AutoModelForImageTextToText.from_pretrained(os.environ['PARTC_MODEL_DIR'], local_files_only=True,
                          dtype=smoke.DTYPE, device_map={'': f'cuda:{rank}'}, attn_implementation='sdpa')
        model.requires_grad_(False)
        model.train()
        parameter = model.model.visual.merger.mlp[-1].weight
        parameter.data = parameter.detach().float()
        parameter.requires_grad_(True)
        original = parameter.detach().cpu().clone()
        stage = shared.TimedStage(model.model.language_model, model.lm_head, rank)
        original_plan = json.loads((DATA/'plan.json').read_text())
        low, high = 'image_count_low', 'image_count_high'
        keys = sorted({key for arm in (low, high) for pair in original_plan[arm] for key in pair})
        inputs = {key: shared.load_input(DATA, key, model) for key in keys}
        length = max(batch[2].shape[-1] for batch in inputs.values())
        padded = {k: pad_batch(v, length, model.config.text_config.pad_token_id or 0) for k,v in inputs.items()}
        plans = {}
        for design in ('single_multi', 'original_high'):
            plans[design] = {}
            for seed in SEEDS:
                plan, counts = make_plan(DATA, design, seed)
                plans[design][seed] = plan
        plans['low_high'] = {}
        for seed in SEEDS:
            rng = random.Random(seed)
            by_arm = {}
            for arm in (low, high):
                pairs = list(original_plan[arm][1:])
                rng.shuffle(pairs)
                by_arm[arm] = pairs
            plans['low_high'][seed] = by_arm
        if rank == 0:
            (OUT/'plans.json').write_text(json.dumps(plans, indent=2)+'\n')

        def reset():
            torch.manual_seed(1234)
            with torch.no_grad():
                parameter.copy_(original)
            parameter.grad = None
            return torch.optim.AdamW([parameter], lr=1e-4, weight_decay=0)

        def step(optimizer, pair, bank, execute=shared.execute_step):
            return execute(stage, model, rank, group, parameter, optimizer, [bank[k] for k in pair])

        warm_pair = original_plan[high][0]
        seen = set()
        for pair in original_plan[high][1:]:
            n = counts[pair[0]]
            if n in seen:
                continue
            seen.add(n)
            opt = reset()
            first = step(opt,pair,inputs)
            grad = parameter.grad.detach().clone().flatten()
            del opt
            opt = reset()
            second = step(opt,pair,padded)
            cosine = F.cosine_similarity(grad, parameter.grad.detach().flatten(), dim=0).item()
            assert cosine >= 0.98, (pair,cosine)
            torch.testing.assert_close(torch.tensor(first['losses']),torch.tensor(second['losses']),rtol=.02,atol=.05)
            if rank == 0:
                append('padding.jsonl', dict(pair=pair, original_length=first['seq_len'], padded_length=length,
                       original_losses=first['losses'], padded_losses=second['losses'], grad_cosine=cosine))
            del opt, grad
        assert seen == {1,2,4,8}

        def run_arm(design, seed, arm, rep, profile=False):
            opt = reset()
            bank = inputs if design == 'low_high' else padded
            initial_pair = original_plan[arm][0] if design == 'low_high' else warm_pair
            step(opt, initial_pair, bank)
            for i,pair in enumerate(plans[design][seed][arm]):
                record = dict(design=design, seed=seed, arm=arm, repeat=rep, step=i+1,
                              ids=pair, images=[counts[k] for k in pair])
                if profile:
                    from traced_step import execute_step
                    from wait_profile import capture_wait
                    with capture_wait() as profiler:
                        row = step(opt,pair,bank,execute_step)
                    local = profiler.result()
                    profiles = [None, None]
                    dist.all_gather_object(profiles,local,group=group)
                    row['ranks'] = profiles
                else:
                    row = step(opt,pair,bank)
                if rank == 0:
                    append('profile.jsonl' if profile else 'time.jsonl', {**record, **row})
            assert torch.isfinite(parameter.grad).all() and parameter.grad.norm().item() > 0
            assert not torch.equal(parameter.detach().cpu(), original)
            del opt
            parameter.grad = None

        for design in ('single_multi','original_high','low_high'):
            arms = (low,high) if design == 'low_high' else ARMS
            for index, seed in enumerate(SEEDS):
                for rep in range(4):
                    ordered = arms if (index + int(rep in (1,2))) % 2 == 0 else arms[::-1]
                    for arm in ordered:
                        run_arm(design,seed,arm,rep+1)
                if rank == 0:
                    print('Timing complete:', design, seed, flush=True)
            if rank == 0:
                write_summary(OUT)

        for index,seed in enumerate(SEEDS):
            for arm in (ARMS if index % 2 == 0 else ARMS[::-1]):
                run_arm('single_multi',seed,arm,1,profile=True)
                if rank == 0:
                    print('GPU trace complete:',seed,arm,flush=True)
                    write_summary(OUT)
    finally:
        smoke.parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == '__main__':
    run()

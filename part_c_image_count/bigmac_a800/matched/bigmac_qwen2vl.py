"""Paired A800 benchmark of the team's Optimus-inspired runtime and the BigMac executor."""
from datetime import timedelta
import importlib.util
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from optimus_runtime import PlanExecutor, Transport
from optimus_schedule import make_plan, rank_tasks

ROOT = Path(os.environ['PARTC_RUN_DIR'])
PRIMARY = Path(os.environ['PARTC_PRIMARY'])
spec = importlib.util.spec_from_file_location('partc_primary', PRIMARY / 'code/bigmac_qwen2vl.py')
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
smoke = shared.smoke
CASES = tuple('image_count_' + level for level in ('low', 'medium', 'high'))


class OptimusAdapter:
    """Use the same HF model and loss, changing only the runtime that schedules the stages."""
    dtype = smoke.DTYPE

    def __init__(self, stage, model, inputs):
        self.stage = stage
        self.visual = model.model.visual
        self.inputs = inputs

    def encoder_shape(self, m):
        return (smoke.VISUAL, smoke.HIDDEN)

    def hidden_shape(self, m):
        return (self.inputs[m][2].shape[-1], 1, smoke.HIDDEN)

    def encoder_forward(self, m):
        pixels, grids = self.inputs[m][:2]
        with torch.autocast('cuda', dtype=self.dtype):
            return self.visual(pixels.cuda().to(self.dtype), grid_thw=grids.cuda()).pooler_output.to(self.dtype).contiguous()

    def llm_forward(self, m, upstream, embeddings):
        _, _, ids, attention, labels, mask, positions = self.inputs[m]
        # The team runtime transfers the embeddings/hidden states, while both decoder
        # stages receive their own small metadata tensors from the same prepared inputs.
        inputs = {'attention': attention.cuda(), 'positions': positions.cuda()}
        if self.stage.pre_process:
            inputs.update(ids=ids.cuda(), image_mask=mask.cuda(), vision_embeds=embeddings)
        else:
            inputs['labels'] = labels.cuda()
            self.stage.set_input_tensor(upstream)
        with torch.autocast('cuda', dtype=self.dtype):
            return self.stage(inputs, None, {})

    def loss(self, m, output):
        # TimedStage uses the identical next-token cross entropy on the last rank.
        return output


def calibrate(transport, stage, model, cases, batches, parameter, rank):
    costs_cache, evidence = {}, {}
    unit = {k: [[1., 1.], [1., 1.]] for k in ('EF', 'EB', 'F', 'B')}
    for ids in cases:
        selected = [batches[key] for key in ids]
        shape = tuple(tuple(v[1].reshape(-1).tolist()) for v in selected)
        if shape in costs_cache:
            continue
        task = PlanExecutor(OptimusAdapter(stage, model, selected), transport, torch.device('cuda',rank))
        measured = task.run(make_plan(2, [0, 0], unit, 'serial'))
        costs = {k: [[None, None], [None, None]] for k in unit}
        for iv in measured['compute_intervals']:
            costs[iv['kind']][iv['rank']][iv['microbatch']] = max((iv['end_s'] - iv['start_s'])*1000, 1e-6)
        for kind in ('EF','EB'):
            costs[kind][1] = costs[kind][0].copy()
        # This is the team's default fixed round-robin placement and measured bubble plan.
        plan = make_plan(2, [0, 1], costs, 'bubble')
        costs_cache[shape] = plan
        evidence[str(shape)] = {'costs_ms':costs, 'plan':plan,
                                'rank_order':[[t.key for t in rank_tasks(plan,r)] for r in range(2)]}
        parameter.grad = None
        stage.pipeline_input = None
    if rank == 0:
        (ROOT/'results/plans.json').write_text(json.dumps(evidence,indent=2)+'\n')
    return costs_cache


def optimus_step(stage, model, rank, group, parameter, optimizer, selected, transport, plans):
    shape = tuple(tuple(v[1].reshape(-1).tolist()) for v in selected)
    backend = PlanExecutor(OptimusAdapter(stage, model, selected), transport, torch.device('cuda',rank))
    plan = plans[shape]
    dist.barrier(group=group)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    losses = backend.run(plan, trace=False)
    dist.all_reduce(parameter.grad, group=group)
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    raw = {'seconds':elapsed,'peak_bytes':peak,'losses':[float(losses[m]) for m in sorted(losses)]}
    gathered = [None,None]
    dist.all_gather_object(gathered,raw,group=group)
    stage.pipeline_input = None
    return {'seconds':max(r['seconds'] for r in gathered), 'peak_bytes':[r['peak_bytes'] for r in gathered],
            'losses':gathered[1]['losses'], 'seq_len':selected[0][2].shape[-1]}


def fingerprint(parameter, sample):
    grad = parameter.grad.detach().flatten()
    weight = parameter.detach().flatten()
    # Keep validation and host copies outside the measured interval in both arms.
    return {'grad':grad[sample].cpu(), 'weight':weight[sample].cpu(),
            'grad_norm':float(torch.linalg.vector_norm(grad))}


def check_replay(stage, model, rank, group, parameter, original, selected, transport, strategies):
    observations = {}
    for label in ('BigMac', 'BigMac repeat', 'Optimus', 'Optimus repeat'):
        with torch.no_grad():
            parameter.copy_(original)
        parameter.grad = None
        torch.manual_seed(1234)
        optimizer = torch.optim.AdamW([parameter], lr=1e-4, weight_decay=0)
        if label.startswith('BigMac'):
            row = shared.execute_step(stage, model, rank, group, parameter, optimizer, selected)
        else:
            row = optimus_step(stage, model, rank, group, parameter, optimizer, selected, transport, strategies)
        observations[label] = (parameter.grad.detach().float().cpu().flatten().clone(),
                               parameter.detach().float().cpu().flatten().clone(), row['losses'])
        del optimizer
    if rank == 0:
        differences = {}
        for left, right in [('BigMac', 'BigMac repeat'), ('Optimus', 'Optimus repeat'), ('BigMac', 'Optimus')]:
            x, wx, losses_x = observations[left]
            y, wy, losses_y = observations[right]
            flip = x*y < 0
            mismatch = (wx - wy).abs() > 5e-6 + 1e-4*wx.abs()
            differences[left + ' vs ' + right] = {
                'losses': [losses_x, losses_y], 'parameters': x.numel(),
                'full_gradient_relative_l2': float((x-y).norm()/x.norm()),
                'full_gradient_cosine': float(torch.nn.functional.cosine_similarity(x.double(),y.double(),dim=0)),
                'full_gradient_max_abs': float((x-y).abs().max()),
                'grad_sign_flips': int(flip.sum()),
                'largest_flipped_grad_abs': max(float(x[flip].abs().max()),float(y[flip].abs().max())) if flip.any() else 0.0,
                'weights_outside_gate': int(mismatch.sum()), 'weight_max_abs': float((wx-wy).abs().max())}
        (ROOT/'results/replay.json').write_text(json.dumps(differences,indent=2)+'\n')
        print('Independent replays and cross-runtime first-step comparison',json.dumps(differences),flush=True)
    parameter.grad = None


def run():
    rank = int(os.environ['LOCAL_RANK'])
    print(f"Worker {rank} imported benchmark", flush=True)
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    device = torch.device('cuda',rank)
    dist.init_process_group('nccl',timeout=timedelta(seconds=110),device_id=device)
    print(f"Worker {rank} initialized NCCL", flush=True)
    smoke.parallel_state.initialize_model_parallel(pipeline_model_parallel_size=2)
    try:
        group = smoke.parallel_state.get_pipeline_model_parallel_group()
        print(f'Worker {rank} loading model', flush=True)
        model = smoke.AutoModelForImageTextToText.from_pretrained(os.environ['PARTC_MODEL_DIR'],
                    local_files_only=True, dtype=smoke.DTYPE, device_map={'':f'cuda:{rank}'},attn_implementation='sdpa')
        model.requires_grad_(False)
        model.train()
        parameter = model.model.visual.merger.mlp[-1].weight
        parameter.data = parameter.detach().float()
        parameter.requires_grad_(True)
        original = parameter.detach().cpu().clone()
        sample = torch.arange(1024,device=device,dtype=torch.int64)*(parameter.numel()-1)//1023
        stage = shared.TimedStage(model.model.language_model,model.lm_head,rank)
        plan = json.loads((PRIMARY/'data/plan.json').read_text())
        assert set(plan) == set(CASES)
        ids = [pair for case in CASES for pair in plan[case]]
        batches = {k:shared.load_input(PRIMARY/'data',k,model) for k in sorted({key for pair in ids for key in pair})}
        print(f"Worker {rank} loaded model and {len(batches)} inputs", flush=True)
        transport = Transport(2,device)
        print(f"Worker {rank} calibrating schedules", flush=True)
        strategies = calibrate(transport,stage,model,ids,batches,parameter,rank)
        print(f'Worker {rank} calibrated; starting timed blocks', flush=True)
        check_replay(stage,model,rank,group,parameter,original,[batches[k] for k in plan[CASES[0]][0]],transport,strategies)
        work = ROOT/'results/steps.jsonl'
        if rank == 0:
            work.touch(exist_ok=False)
        low, med, high = CASES
        order = ((low,med,high),(high,med,low),(med,high,low),(low,high,med))
        for b,case_order in enumerate(order):
            for c,case in enumerate(case_order):
                assert len(plan[case])==9 and len(set(sum(plan[case][1:],[])))==12
                # Each named condition starts on the same backend twice and finishes on it twice.
                systems = ('BigMac','Optimus-PP') if b % 2 == 0 else ('Optimus-PP','BigMac')
                reference = []
                for first,system in enumerate(systems):
                    torch.manual_seed(1234)
                    with torch.no_grad():
                        parameter.copy_(original)
                    parameter.grad = None
                    optimizer = torch.optim.AdamW([parameter],lr=1e-4,weight_decay=0)
                    for i,pair in enumerate(plan[case]):
                        inputs = [batches[key] for key in pair]
                        if system == 'BigMac':
                            row = shared.execute_step(stage,model,rank,group,parameter,optimizer,inputs)
                        else:
                            row = optimus_step(stage,model,rank,group,parameter,optimizer,inputs,transport,strategies)
                        snap = fingerprint(parameter,sample)
                        assert snap['grad_norm'] > 0 and math.isfinite(snap['grad_norm'])
                        assert row['seconds'] > 0 and all(math.isfinite(v) for v in row['losses'])
                        if not first:
                            reference.append((row['losses'],snap))
                        else:
                            expected_loss, expected = reference[i]
                            # BF16 computes both arms, with identical FP32 gradient summation and AdamW.
                            torch.testing.assert_close(torch.tensor(row['losses']),torch.tensor(expected_loss),rtol=1e-3,atol=2e-4)
                            torch.testing.assert_close(snap['weight'],expected['weight'],rtol=1e-4,atol=5e-6)
                            torch.testing.assert_close(snap['grad'],expected['grad'],rtol=1e-2,atol=2e-5)
                            assert math.isclose(snap['grad_norm'],expected['grad_norm'],rel_tol=2e-3)
                        if rank==0:
                            record = dict(system=system,case=case,block=b+1,step=i,warmup=i==0,ids=pair,
                                          grad_norm=snap['grad_norm'],checked_against_other=bool(first),**row)
                            with work.open('a') as f:
                                f.write(json.dumps(record)+'\n')
                    assert not torch.equal(parameter.detach().cpu(),original)
                    del optimizer
                    parameter.grad = None
                    if rank == 0:
                        print('Complete',b+1,case,system,flush=True)
        if rank == 0:
            (ROOT/'results/numeric_pass.json').write_text(json.dumps({'steps_compared_per_rank':108,'ranks':2,
                'sampled_gradient_and_weight_coordinates_per_step':1024,'both_losses_per_step':True,
                'source':'1ca3e2a15ae83b6a86dba802b59153b45ce8865e'})+'\n')
    finally:
        smoke.parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == '__main__':
    run()

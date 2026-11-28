"""Profile-only copy of the original BigMac timed step with trace boundary marks."""
from bigmac_shared import *
from wait_profile import mark_start, mark_end

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
    mark_start()
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    executor.execute()
    dist.all_reduce(parameter.grad, group=group)
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    mark_end()
    peak = torch.cuda.max_memory_allocated()
    losses = [bridge.language[i]['loss'].item() for i in range(2)] if rank == 1 else []
    gathered = [None, None]
    dist.all_gather_object(gathered, {'seconds': elapsed, 'peak_bytes': peak, 'losses': losses}, group=group)
    assert all(math.isfinite(value) for value in gathered[1]['losses'])
    stage.pipeline_input = None
    return {'seconds': max(item['seconds'] for item in gathered), 'peak_bytes': [g['peak_bytes'] for g in gathered],
            'losses': gathered[1]['losses'], 'seq_len': length}

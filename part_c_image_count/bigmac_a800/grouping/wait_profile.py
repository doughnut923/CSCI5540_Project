"""Local-device activity in a synchronized BigMac step. All reported times are ms."""
import time
from contextlib import contextmanager
from contextvars import ContextVar


_current = ContextVar("partc_profile", default=None)
_unavailable_reason = None


def _covered(intervals):
    total = 0
    last_end = -1
    for start, end in sorted(intervals):
        total += max(0, end - max(last_end, start))
        last_end = max(last_end, end)
    return total


def _summarize(intervals, start, end):
    clipped = []
    noncomm = []
    nccl_count = 0
    for left, right, is_nccl in intervals:
        left, right = max(start, left), min(end, right)
        if left >= right:
            continue
        clipped.append((left, right))
        if is_nccl:
            nccl_count += 1
        else:
            noncomm.append((left, right))
    if not noncomm or not nccl_count:
        raise RuntimeError(f"Incomplete GPU trace: {len(noncomm)} non-NCCL and {nccl_count} NCCL events in step")
    busy = _covered(clipped)
    work = _covered(noncomm)
    return dict(available=True, source="kineto_cuda", gpu_idle_ms=(end - start - busy) / 1e6,
                gpu_busy_ms=busy / 1e6, noncomm_busy_ms=work / 1e6,
                nccl_only_ms=(busy - work) / 1e6, gpu_events=len(clipped),
                nccl_events=nccl_count, error=None)


class _Capture:
    def __init__(self):
        self.start = self.end = None
        self.start_mono = self.end_mono = None
        self.wait_events = []
        self.stats = dict(available=False, source="unavailable", window_ms=None,
                          gpu_idle_ms=None, gpu_busy_ms=None, noncomm_busy_ms=None,
                          nccl_only_ms=None, gpu_events=None, nccl_events=None,
                          explicit_work_wait_ms=None, explicit_wait_calls=None, error=None)

    def result(self):
        return self.stats.copy()


def mark_start():
    p = _current.get()
    if p is not None:
        p.start_mono = time.monotonic_ns()
        p.start = time.time_ns()


def mark_end():
    p = _current.get()
    if p is not None:
        p.end = time.time_ns()
        p.end_mono = time.monotonic_ns()


def _read_trace(prof, device, p, torch):
    # Kineto returns Unix epoch nanoseconds, unlike its relative Chrome-export ticks.
    trace = prof.profiler.kineto_results
    lead = p.start - trace.trace_start_ns()
    if not 0 <= lead < 120_000_000_000:
        raise RuntimeError("Host timestamps do not match Kineto trace clock")
    intervals = []
    for evt in trace.events():
        if evt.device_type() != torch.autograd.DeviceType.CUDA or evt.device_index() != device:
            continue
        if evt.is_user_annotation():
            continue
        name = evt.name().lower()
        intervals.append((evt.start_ns(), evt.end_ns(), "nccl" in name))
    return _summarize(intervals, p.start, p.end)


def _install_work_wait(p, torch):
    # A fallback only: stream time crossing an explicit torch.distributed.Work.wait.
    # It can include host launch gaps. Durations are summed per call, across streams.
    work_class = torch.distributed.Work
    original = work_class.wait

    def measured(work, *args, **kwargs):
        if p.start is None or p.end is not None:
            return original(work, *args, **kwargs)
        stream = torch.cuda.current_stream()
        before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        before.record(stream)
        result = original(work, *args, **kwargs)
        after.record(stream)
        p.wait_events.append((before, after))
        return result

    work_class.wait = measured
    return work_class, original


@contextmanager
def capture_wait():
    import torch

    global _unavailable_reason
    p = _Capture()
    token = _current.set(p)
    prof = hook = None
    try:
        if _unavailable_reason is None:
            try:
                cuda = torch.profiler.ProfilerActivity.CUDA
                if cuda not in torch.profiler.supported_activities():
                    raise RuntimeError("CUDA profiling unavailable")
                prof = torch.profiler.profile(activities=[cuda])
                prof.start()
            except Exception as exc:
                _unavailable_reason = f"{type(exc).__name__}: {exc}"
                prof = None
        if prof is None:
            p.stats["error"] = _unavailable_reason
            try:
                hook = _install_work_wait(p, torch)
            except (AttributeError, TypeError, RuntimeError) as exc:
                p.stats["error"] += f"; CUDA event fallback unavailable: {exc}"
        try:
            yield p
        finally:
            if hook is not None:
                work_class, original = hook
                work_class.wait = original
            if prof is not None:
                try:
                    prof.stop()
                except Exception as exc:
                    p.stats["error"] = _unavailable_reason = f"{type(exc).__name__}: {exc}"
                    prof = None
        if p.start is None or p.end is None or p.end <= p.start:
            p.stats["error"] = "Missing or invalid step boundary marks"
            return
        p.stats["window_ms"] = (p.end_mono - p.start_mono) / 1e6
        if abs((p.end - p.start) / 1e6 - p.stats["window_ms"]) > 0.5:
            p.stats["error"] = "Host wall clock changed in the step"
            return
        if prof is not None:
            try:
                p.stats.update(_read_trace(prof, torch.cuda.current_device(), p, torch))
            except Exception as exc:
                p.stats["error"] = _unavailable_reason = f"{type(exc).__name__}: {exc}"
        elif p.wait_events:
            try:
                p.stats.update(source="explicit_work_wait_cuda_events",
                               explicit_wait_calls=len(p.wait_events),
                               explicit_work_wait_ms=sum(left.elapsed_time(right) for left, right in p.wait_events))
            except RuntimeError as exc:
                p.stats["error"] += f"; CUDA event read failed: {exc}"
    finally:
        _current.reset(token)

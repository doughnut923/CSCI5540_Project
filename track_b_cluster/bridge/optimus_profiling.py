"""Opt-in synchronized wall-time diagnostics; not an overlap benchmark."""
from contextlib import contextmanager
import math
import time


TOP_LEVEL_PHASES = ("model_schedule", "ordinary_grad_collect", "encoder_grad_sync",
                    "tied_grad_sync", "global_grad_norm", "optimizer")


class StepProfiler:
    def __init__(self, enabled=False, synchronize=None, clock=time.monotonic):
        self.enabled = enabled
        self.synchronize = synchronize
        self.clock = clock
        self.times = {}

    @contextmanager
    def phase(self, name):
        if not self.enabled:
            yield
            return
        start = self.clock()
        try:
            yield
        finally:
            if self.synchronize is not None:
                self.synchronize()
            elapsed = self.clock() - start
            self.times[name] = self.times.get(name, 0.0) + elapsed

    def breakdown(self, step_time):
        """Disjoint top-level times; optimizer children remain separate."""
        if not self.enabled:
            return None
        if not math.isfinite(step_time) or step_time <= 0:
            raise ValueError("step time must be finite and positive")
        if any(not math.isfinite(v) or v < 0 for v in self.times.values()):
            raise ValueError("invalid phase duration")
        phases = {name: self.times.get(name, 0.0) for name in TOP_LEVEL_PHASES}
        remaining = step_time - sum(phases.values())
        if remaining < -max(1e-6, step_time * 1e-6):
            raise ValueError("top-level phases overlap or exceed the complete step")
        phases["unattributed"] = max(0.0, remaining)
        children = sum(value for key, value in self.times.items() if key.startswith("optimizer."))
        if children > phases["optimizer"] + max(1e-6, step_time * 1e-6):
            raise ValueError("optimizer children exceed their parent duration")
        return phases

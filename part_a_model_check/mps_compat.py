"""Shape-preserving workaround for the observed PyTorch 2.6 MPS scatter bug."""

from torch.overrides import TorchFunctionMode


class FlattenMPSScatterSource(TorchFunctionMode):
    """Pass an explicit flat view, preserving the original source and autograd view."""

    def __init__(self):
        super().__init__()
        self.applied_calls = 0

    def __torch_function__(self, function, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if (function.__name__ == "masked_scatter" and len(args) >= 3
                and args[0].device.type == "mps" and args[2].ndim > 1):
            source = args[2].flatten()
            args = (*args[:2], source, *args[3:])
            self.applied_calls += 1
        return function(*args, **kwargs)

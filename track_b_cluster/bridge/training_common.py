"""Shared optimizer from the team DistTrain runner.

The default step keeps the existing per-unit clipping behavior. The colocated
runner supplies one whole-model norm so encoder replicas are counted once.
"""
import math

import torch


class MainGradAdamW:
    """AdamW over MCore's fp32 main_grad buffers, with fp32 master weights.

    The fused path accumulates into `param.main_grad` and never writes `.grad`, so the
    fp32 masters take those buffers as their grads and are cast back into the bf16 params
    after every update (the same split Megatron's mixed-precision optimizer uses).
    """

    def __init__(self, model, args, total_steps):
        self.params = [p for p in model.parameters()
                       if getattr(p, "main_grad", None) is not None]
        self.masters = [p.detach().float().clone().requires_grad_(True) for p in self.params]
        self.opt = torch.optim.AdamW(
            self.masters, lr=args.lr, betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps, weight_decay=args.weight_decay)
        self.lr, self.min_lr = args.lr, args.min_lr
        self.clip = args.clip_grad
        self.total = max(1, total_steps)
        self.steps = 0

    def step(self, global_grad_norm=None):
        for master, p in zip(self.masters, self.params):
            master.grad = p.main_grad
        lr = self.min_lr + 0.5 * (self.lr - self.min_lr) * (
            1.0 + math.cos(math.pi * self.steps / self.total))
        for group in self.opt.param_groups:
            group["lr"] = lr
        if not self.masters:
            raise ValueError("optimizer has no trainable main_grad parameters")
        if global_grad_norm is None:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.masters, self.clip))
        else:
            grad_norm = float(global_grad_norm)
            if not math.isfinite(grad_norm):
                raise FloatingPointError("non-finite global gradient norm")
            scale = min(1.0, self.clip / (grad_norm + 1e-6))
            for master in self.masters:
                master.grad.mul_(scale)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for master, p in zip(self.masters, self.params):
                p.data.copy_(master)
        self.steps += 1
        return {"lr": lr, "grad_norm": grad_norm,
                "weight_norm": float(self.masters[0].norm())}

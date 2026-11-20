"""Execute a validated PP plan with real autograd and point-to-point transfers.

The model adapter supplies the team's model/data semantics. Sends keep tensors
alive until completion; separate directional process groups avoid head-of-line
blocking between encoder and LLM traffic. No global barrier is inserted per op.
"""
import time

import torch
import torch.distributed as dist

from optimus_schedule import rank_tasks, validate_plan


def sample_indices(length, count, device):
    """Integer arithmetic avoids FP32 rounding beyond a large tensor's end."""
    count = min(length, count)
    if length < 1 or count < 1:
        return torch.empty(0, dtype=torch.int64, device=device)
    return torch.arange(count, dtype=torch.int64, device=device) * (length - 1) // max(1, count - 1)


class Transport:
    def __init__(self, world, device):
        self.device = device
        self.groups = {}
        self.pending = []
        # All ranks must construct groups in the same order, including outsiders.
        for channel, pairs in (
            ("forward", [(r, r + 1) for r in range(world - 1)]),
            ("backward", [(r, r + 1) for r in range(world - 1)]),
            ("encoder", [(0, r) for r in range(1, world)]),
            ("encoder_grad", [(0, r) for r in range(1, world)]),
        ):
            for pair in pairs:
                self.groups[(channel, pair)] = dist.new_group(list(pair))

    def group(self, channel, a, b):
        return self.groups[(channel, tuple(sorted((a, b))))]

    def send(self, tensor, peer, channel):
        payload = tensor.detach().contiguous()
        handle = dist.isend(payload, peer, group=self.group(channel, dist.get_rank(), peer))
        self.pending.append((handle, payload))

    def recv(self, shape, dtype, peer, channel):
        tensor = torch.empty(shape, dtype=dtype, device=self.device)
        dist.irecv(tensor, peer, group=self.group(channel, dist.get_rank(), peer)).wait()
        return tensor

    def finish(self):
        for handle, _ in self.pending:
            handle.wait()
        self.pending.clear()


class PlanExecutor:
    def __init__(self, adapter, transport, device):
        self.adapter = adapter
        self.transport = transport
        self.device = device
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()

    def run(self, plan, capture_image_grads=False):
        validate_plan(plan)
        if plan["world"] != self.world:
            raise ValueError("plan world differs from process group")
        owners = plan["owners"]
        encoded, embeddings, inputs, outputs, losses = {}, {}, {}, {}, {}
        marks, compute_marks = [], []
        image_grad_norms = {}
        cuda = self.device.type == "cuda"
        dist.barrier()
        if cuda:
            torch.cuda.synchronize()
            origin = torch.cuda.Event(enable_timing=True)
            origin.record()
        t0 = time.monotonic()
        def compute(task, fn):
            start = torch.cuda.Event(enable_timing=True) if cuda else time.monotonic()
            if cuda:
                start.record()
            value = fn()
            end = torch.cuda.Event(enable_timing=True) if cuda else time.monotonic()
            if cuda:
                end.record()
            compute_marks.append((task, start, end))
            return value

        for task in rank_tasks(plan, self.rank):
            m, kind = task.microbatch, task.kind
            start = torch.cuda.Event(enable_timing=True) if cuda else time.monotonic()
            if cuda:
                start.record()
            if kind == "EF":
                encoded[m] = compute(task, lambda: self.adapter.encoder_forward(m))
                if (tuple(encoded[m].shape) != tuple(self.adapter.encoder_shape(m))
                        or encoded[m].dtype != self.adapter.dtype):
                    raise ValueError("encoder output does not match transfer shape/dtype")
                if self.rank == 0:
                    embeddings[m] = encoded[m].detach().requires_grad_(True)
                else:
                    self.transport.send(encoded[m], 0, "encoder")
            elif kind == "F":
                if self.rank == 0:
                    if owners[m] != 0:
                        embeddings[m] = self.transport.recv(
                            self.adapter.encoder_shape(m), self.adapter.dtype,
                            owners[m], "encoder").requires_grad_(True)
                    inp = None
                else:
                    inp = self.transport.recv(self.adapter.hidden_shape(m),
                                              self.adapter.dtype, self.rank - 1,
                                              "forward").requires_grad_(True)
                inputs[m] = inp
                def forward():
                    output = self.adapter.llm_forward(m, inp, embeddings.get(m))
                    if self.rank == self.world - 1:
                        losses[m] = self.adapter.loss(m, output)
                    return output
                outputs[m] = compute(task, forward)
                if self.rank < self.world - 1:
                    if (tuple(outputs[m].shape) != tuple(self.adapter.hidden_shape(m))
                            or outputs[m].dtype != self.adapter.dtype):
                        raise ValueError("LLM output does not match transfer shape/dtype")
                    self.transport.send(outputs[m], self.rank + 1, "forward")
            elif kind == "B":
                if self.rank == self.world - 1:
                    # The mean of microbatch losses matches the team native runner.
                    compute(task, lambda: (losses[m] / len(owners)).backward())
                else:
                    grad = self.transport.recv(outputs[m].shape, outputs[m].dtype,
                                               self.rank + 1, "backward")
                    compute(task, lambda: torch.autograd.backward(outputs[m], grad))
                if self.rank > 0:
                    if inputs[m].grad is None:
                        raise RuntimeError("missing pipeline input gradient")
                    self.transport.send(inputs[m].grad, self.rank - 1, "backward")
                else:
                    grad = embeddings[m].grad
                    if grad is None:
                        raise RuntimeError("missing encoder output gradient")
                    if capture_image_grads:
                        ranges = self.adapter.encoder_gradient_ranges(m)
                        image_grad_norms[str(m)] = [float(torch.linalg.vector_norm(grad[a:b].float()))
                                                   for a, b in ranges]
                    if owners[m] == 0:
                        embeddings[m] = grad
                    else:
                        self.transport.send(grad, owners[m], "encoder_grad")
                        del embeddings[m]
                del outputs[m], inputs[m]
            elif kind == "EB":
                grad = (embeddings.pop(m) if self.rank == 0 else self.transport.recv(
                    encoded[m].shape, encoded[m].dtype, 0, "encoder_grad"))
                output = encoded.pop(m)
                compute(task, lambda: torch.autograd.backward(output, grad))
            else:
                raise ValueError("unknown task kind " + kind)
            end = torch.cuda.Event(enable_timing=True) if cuda else time.monotonic()
            if cuda:
                end.record()
            marks.append((task, start, end))
        self.transport.finish()
        if encoded or embeddings or inputs or outputs:
            raise RuntimeError("unconsumed activation state at iteration boundary")
        if cuda:
            torch.cuda.synchronize()
        elapsed = time.monotonic() - t0
        def intervals(source):
            result = []
            for task, start, end in source:
                begin = origin.elapsed_time(start) / 1000 if cuda else start - t0
                finish = origin.elapsed_time(end) / 1000 if cuda else end - t0
                result.append({"kind": task.kind, "rank": self.rank,
                               "microbatch": task.microbatch,
                               "start_s": begin, "end_s": finish})
            return result
        # EF/F/B/EB ranges include their transfer/wait time. They are execution
        # ranges, not kernel-busy intervals; do not call their complement bubbles.
        local = {"rank": self.rank, "elapsed_s": elapsed, "intervals": intervals(marks),
                 "compute": intervals(compute_marks),
                 "image_grad_norms": image_grad_norms,
                 "losses": {str(m): float(v.detach()) for m, v in losses.items()}}
        gathered = [None] * self.world
        dist.all_gather_object(gathered, local)
        result = {"step_time_s": max(r["elapsed_s"] for r in gathered),
                "loss": sum(gathered[-1]["losses"].values()) / len(owners),
                "execution_intervals": [v for r in gathered for v in r["intervals"]],
                "compute_intervals": [v for r in gathered for v in r["compute"]]}
        result["image_embedding_grad_norms"] = gathered[0]["image_grad_norms"]
        return result


def collect_main_grads(module):
    """Bridge ordinary autograd grads as well as TE's fused main_grad path."""
    for p in module.parameters():
        if p.requires_grad and p.grad is not None:
            if getattr(p, "grad_added_to_main_grad", False):
                # TE may return a dummy .grad after its fused kernel has already
                # accumulated the real weight gradient into main_grad.
                pass
            elif getattr(p, "main_grad", None) is None:
                p.main_grad = p.grad.detach().float().clone()
            elif p.grad is not p.main_grad:
                p.main_grad.add_(p.grad.detach().float())
            p.grad = None


def zero_grads(module):
    for p in module.parameters():
        p.grad = None
        if hasattr(p, "grad_added_to_main_grad"):
            p.grad_added_to_main_grad = False
        if getattr(p, "main_grad", None) is not None:
            p.main_grad.zero_()


def sync_encoder_grads(encoder):
    # Each leaf gradient already includes 1/global_microbatches. Sum, don't
    # average by replica count: replicas process disjoint microbatches.
    for p in encoder.parameters():
        if p.requires_grad:
            dist.all_reduce(p.main_grad, op=dist.ReduceOp.SUM)


def global_grad_norm(encoder, language):
    device = next(language.parameters()).device
    total = torch.zeros((), dtype=torch.float64, device=device)
    for p in language.parameters():
        # Megatron marks the duplicate tied output embedding as shared.
        if p.requires_grad and not getattr(p, "shared", False):
            total += torch.linalg.vector_norm(p.main_grad).double().square()
    if dist.get_rank() == 0:
        for p in encoder.parameters():
            if p.requires_grad:
                total += torch.linalg.vector_norm(p.main_grad).double().square()
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return math_sqrt(total)


def math_sqrt(tensor):
    return float(tensor.sqrt())

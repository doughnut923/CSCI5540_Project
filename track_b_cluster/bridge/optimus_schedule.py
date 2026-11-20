"""Profile-guided, whole-encoder PP subset of Optimus (not full ATC Optimus).

No torch dependency. Durations are supplied measurements or explicitly labelled
estimates. LLM 1F1B order is fixed; encoder operations may fill its idle intervals.
Every returned plan is checked for dependency and single-device resource safety.
"""
import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Task:
    kind: str
    rank: int
    microbatch: int

    @property
    def key(self):
        return "%s:%d:%d" % (self.kind, self.rank, self.microbatch)


def llm_order(rank, world, microbatches):
    warmup = min(world - rank - 1, microbatches)
    order = [Task("F", rank, m) for m in range(warmup)]
    for m in range(microbatches - warmup):
        order += [Task("F", rank, m + warmup), Task("B", rank, m)]
    order += [Task("B", rank, m) for m in range(microbatches - warmup, microbatches)]
    return order


def dependencies(world, owners):
    deps = {}
    for m, owner in enumerate(owners):
        ef, eb = Task("EF", owner, m), Task("EB", owner, m)
        deps[ef] = set()
        deps[eb] = {Task("B", 0, m)}
        for r in range(world):
            f, b = Task("F", r, m), Task("B", r, m)
            deps[f] = {ef if r == 0 else Task("F", r - 1, m)}
            deps[b] = {f}
            if r + 1 < world:
                deps[b].add(Task("B", r + 1, m))
    # Preserve both LLM order and each encoder replica's send order. The latter
    # keeps Gloo/NCCL point-to-point ordering identical at sender and receiver.
    for r in range(world):
        order = llm_order(r, world, len(owners))
        for previous, task in zip(order, order[1:]):
            deps[task].add(previous)
        for kind in ("EF", "EB"):
            seq = [Task(kind, r, m) for m, owner in enumerate(owners) if owner == r]
            for previous, task in zip(seq, seq[1:]):
                deps[task].add(previous)
    return deps


def validate_costs(costs, world, microbatches):
    for kind in ("EF", "EB", "F", "B"):
        rows = costs[kind]
        if len(rows) != world or any(len(row) != microbatches for row in rows):
            raise ValueError("%s costs must have shape [world, microbatches]" % kind)
        if any(not math.isfinite(v) or v <= 0 for row in rows for v in row):
            raise ValueError("costs must be finite positive durations")


def make_plan(world, owners, costs, mode="bubble"):
    if world < 1 or not owners or any(r < 0 or r >= world for r in owners):
        raise ValueError("invalid world size or encoder owner")
    if mode not in ("serial", "bubble", "coarse"):
        raise ValueError("unknown schedule mode")
    validate_costs(costs, world, len(owners))
    deps = dependencies(world, owners)
    if mode in ("serial", "coarse"):
        # A conservative reference: all encoder forwards precede all LLM work,
        # and all encoder backwards follow it. Coarse can use multiple replicas.
        efs = {t for t in deps if t.kind == "EF"}
        lbs = {t for t in deps if t.kind == "B"}
        for task in deps:
            if task.kind == "F":
                deps[task].update(efs)
            elif task.kind == "EB":
                deps[task].update(lbs)
    pending = set(deps)
    done, running, ends = set(), {}, {}
    events = []
    now = 0.0
    while pending or running:
        for rank, (task, end) in list(running.items()):
            if end <= now + 1e-10:
                done.add(task)
                ends[task] = end
                del running[rank]
        for rank in range(world):
            if rank in running:
                continue
            ready = [t for t in pending if t.rank == rank and deps[t] <= done]
            if not ready:
                continue
            # Run ready LLM work first; free completed encoder graphs next; only
            # then use a waiting stage for its next encoder forward.
            priority = {"B": 0, "F": 1, "EB": 2, "EF": 3}
            task = min(ready, key=lambda t: (priority[t.kind], t.microbatch))
            duration = costs[task.kind][rank][task.microbatch]
            end = now + duration
            pending.remove(task)
            running[rank] = (task, end)
            events.append(dict(asdict(task), start=now, end=end))
        if running:
            now = min(end for _, end in running.values())
        elif pending:
            raise ValueError("dependency cycle in schedule")
    plan = {"mode": mode, "world": world, "owners": list(owners),
            "events": events, "estimated_time_ms": max(ends.values())}
    validate_plan(plan)
    return plan


def validate_plan(plan):
    world, owners = plan["world"], plan["owners"]
    expected = dependencies(world, owners)
    indexed = {}
    for event in plan["events"]:
        task = Task(event["kind"], event["rank"], event["microbatch"])
        if task in indexed or task not in expected:
            raise ValueError("duplicate or unexpected operation: " + task.key)
        if (not math.isfinite(event["start"]) or not math.isfinite(event["end"])
                or event["start"] < 0 or event["end"] <= event["start"]):
            raise ValueError("invalid operation interval")
        indexed[task] = event
    if set(indexed) != set(expected):
        raise ValueError("missing operations")
    for task, requirements in expected.items():
        for previous in requirements:
            if indexed[previous]["end"] > indexed[task]["start"] + 1e-8:
                raise ValueError("dependency violation: " + task.key)
    for rank in range(world):
        events = sorted((e for e in plan["events"] if e["rank"] == rank),
                        key=lambda e: e["start"])
        if any(a["end"] > b["start"] + 1e-8 for a, b in zip(events, events[1:])):
            raise ValueError("overlapping compute on one rank")


def choose_plan(world, microbatches, costs):
    """Compare coarse schedules and bubble schedules on a bounded placement set.

    This is a placement heuristic, not the paper's full 3D-parallel-plan search.
    No sample is shuffled: owners choose where each fixed microbatch is encoded.
    """
    candidates = {tuple([0] * microbatches)}
    for shift in range(world):
        candidates.add(tuple((m + shift) % world for m in range(microbatches)))
    # Greedy whole-encoder load balance; all costs still come from the profile.
    loads, owners = [0.0] * world, []
    for m in range(microbatches):
        owner = min(range(world), key=lambda r: (loads[r] + costs["EF"][r][m]
                                                + costs["EB"][r][m], r))
        owners.append(owner)
        loads[owner] += costs["EF"][owner][m] + costs["EB"][owner][m]
    candidates.add(tuple(owners))
    plans = [make_plan(world, owners, costs, mode)
             for owners in sorted(candidates) for mode in ("coarse", "bubble")]
    best = min(plans, key=lambda p: (p["estimated_time_ms"], p["owners"], p["mode"]))
    best["candidates_evaluated"] = len(plans)
    best["implementation"] = "optimus-inspired-pp-whole-encoder"
    return best


def rank_tasks(plan, rank):
    return [Task(e["kind"], rank, e["microbatch"])
            for e in sorted(plan["events"], key=lambda e: (e["start"], e["rank"]))
            if e["rank"] == rank]

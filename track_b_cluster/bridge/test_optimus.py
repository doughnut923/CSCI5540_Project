"""CPU correctness tests; these do not measure CUDA/3090 performance."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from datetime import timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from optimus_runtime import (PlanExecutor, Transport, collect_main_grads,
                             global_grad_norm, sample_indices, sync_encoder_grads, zero_grads)
from optimus_schedule import choose_plan, make_plan, validate_plan
from optimus_optimizer import MainGradAdamW
from manifest_data import ManifestBatches
from optimus_config import model_arguments
from optimus_experiment import (OptimusBatches, make_input_plan, check_step_inputs,
                                interleave_input_plan)


def costs(world, count):
    return {k: [[float((m % 3) + 1) if k.startswith("E") else 2.0
                 for m in range(count)] for _ in range(world)]
            for k in ("EF", "EB", "F", "B")}


def modules(world):
    torch.manual_seed(31)
    return (torch.nn.Sequential(torch.nn.Linear(4, 7), torch.nn.Tanh(), torch.nn.Linear(7, 4)),
            [torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.Tanh())]
            + [torch.nn.Sequential(torch.nn.Linear(6, 6), torch.nn.Tanh()) for _ in range(world - 2)]
            + [torch.nn.Linear(6, 3)])


class ToyAdapter:
    dtype = torch.float32

    def __init__(self, encoder, language, rank, data):
        self.encoder, self.language, self.rank, self.data = encoder, language, rank, data

    def encoder_shape(self, m):
        return (self.data[m][0].shape[0], 4)

    def hidden_shape(self, m):
        return (self.data[m][0].shape[0], 6)

    def encoder_gradient_ranges(self, m):
        return [(i, i + 1) for i in range(self.data[m][0].shape[0])]

    def accumulate_parameter_grads(self, kind):
        collect_main_grads(self.encoder if kind == "EB" else self.language)

    def encoder_forward(self, m):
        return self.encoder(self.data[m][0])

    def llm_forward(self, m, hidden, embedding):
        return self.language(embedding if self.rank == 0 else hidden)

    def loss(self, m, output):
        return (output - self.data[m][1]).square().mean()


def worker(rank, path, result_dir, world):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + path, rank=rank,
                            world_size=world, timeout=timedelta(seconds=60))
    transport = Transport(world, torch.device("cpu"))
    enc, stages = modules(world)
    ref_enc, ref_stages = deepcopy(enc), deepcopy(stages)
    joint = torch.nn.ModuleList([enc, stages[rank]])
    reference = torch.nn.ModuleList([ref_enc] + ref_stages)
    for module in (joint, reference):
        for p in module.parameters():
            p.main_grad = torch.zeros_like(p)
    args = SimpleNamespace(lr=.01, min_lr=.001, adam_beta1=.9, adam_beta2=.95,
                           adam_eps=1e-8, weight_decay=.01, clip_grad=.3)
    optimizer = MainGradAdamW(joint, args, 3)
    ref_opt = MainGradAdamW(reference, args, 3)
    torch.manual_seed(64)
    data = [(torch.randn(m % 3 + 1, 4), torch.randn(m % 3 + 1, 3)) for m in range(5)]
    # Unequal ownership, a rank with no encoder work, and profile-selected work.
    plans = [make_plan(world, [0] * 5, costs(world, 5), "serial"),
             make_plan(world, [0] + [world - 1] * 4, costs(world, 5)),
             choose_plan(world, 5, costs(world, 5))]
    max_error = 0.0
    for plan in plans:
        zero_grads(joint)
        zero_grads(reference)
        reference_network = torch.nn.Sequential(ref_enc, *ref_stages)
        expected = sum((reference_network(x) - y).square().mean() for x, y in data) / len(data)
        expected.backward()
        collect_main_grads(reference)
        adapter = ToyAdapter(enc, stages[rank], rank, data)
        result = PlanExecutor(adapter, transport, torch.device("cpu")).run(plan, capture_image_grads=True)
        for m, norms in result["image_embedding_grad_norms"].items():
            assert len(norms) == data[int(m)][0].shape[0]
            assert all(value > 0 for value in norms)
        collect_main_grads(joint)
        sync_encoder_grads(enc)
        assert abs(result["loss"] - float(expected)) < 1e-6
        for actual, wanted in zip(enc.parameters(), ref_enc.parameters()):
            torch.testing.assert_close(actual.main_grad, wanted.main_grad, rtol=2e-5, atol=1e-6)
        for actual, wanted in zip(stages[rank].parameters(), ref_stages[rank].parameters()):
            torch.testing.assert_close(actual.main_grad, wanted.main_grad, rtol=2e-5, atol=1e-6)
        norm = global_grad_norm(enc, stages[rank])
        expected_norm = sum(p.main_grad.double().square().sum() for p in reference.parameters()).sqrt()
        assert abs(norm - float(expected_norm)) < 1e-6
        optimizer.step(global_grad_norm=norm)
        ref_opt.step(global_grad_norm=float(expected_norm))
        for actual_module, wanted_module in ((enc, ref_enc), (stages[rank], ref_stages[rank])):
            for actual, wanted in zip(actual_module.parameters(), wanted_module.parameters()):
                torch.testing.assert_close(actual, wanted, rtol=2e-5, atol=1e-6)
                max_error = max(max_error, float((actual - wanted).abs().max()))
    Path(result_dir, "rank%d.json" % rank).write_text(json.dumps(
        {"rank": rank, "optimizer_updates": optimizer.steps, "max_parameter_error": max_error}))
    dist.destroy_process_group()


class PlanTests(unittest.TestCase):
    def test_model_overrides_are_local_to_optimus(self):
        from pp2_equivalence import mcore_argv
        with patch.dict("os.environ", {"QWEN2VL_HF_DIR": "unused"}):
            before = mcore_argv(1024)
            ours = model_arguments(1024)
            self.assertEqual(before, mcore_argv(1024))
        self.assertEqual(ours[ours.index("--patch-tokenizer-type") + 1], "Qwen2VLTokenizer")
        self.assertEqual(before[before.index("--patch-tokenizer-type") + 1], "Qwen2Tokenizer")
        self.assertEqual(ours[ours.index("--seq-length") + 1], "1024")

    def test_bucket_plan_requires_complete_coverage(self):
        counts = [1, 2, 4, 8] * 3
        fixture = {"samples": [{"sample_id": str(i), "images": [{}] * count,
                                 "derived_visual_tokens_total": 576}
                                for i, count in enumerate(counts)]}
        audit = {"passed": True, "samples": [
            {"sample_id": str(i), "image_count": count, "sequence_length": 605 + 2 * count}
            for i, count in enumerate(counts)]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "manifest.json")
            path.write_text(json.dumps(fixture))
            source = ManifestBatches("unused", path, tmp, batching="bucket")
            with self.assertRaisesRegex(ValueError, "misses 2 samples"):
                make_input_plan(source, path, audit, 2, 1, 6)
            plan = make_input_plan(source, path, audit, 2, 1, 8)
            self.assertEqual(plan["distinct_measured_samples"], 12)
            for step in plan["steps"]:
                self.assertEqual(len({row["image_count"] for row in step["input_observations"]}), 1)
            mixed = interleave_input_plan(plan)
            self.assertEqual(mixed["sample_multiset_sha256"], plan["sample_multiset_sha256"])
            self.assertNotEqual(mixed["sample_order_sha256"], plan["sample_order_sha256"])
            replay = OptimusBatches("unused", path, tmp, batching="manifest", input_plan=mixed)
            for step in mixed["steps"]:
                self.assertEqual([sample["sample_id"] for sample in replay.plan(step["step"], 2)],
                                 step["sample_ids"])
            wrong = [{"sample_id": "incorrect"}]
            with self.assertRaisesRegex(ValueError, "sample order"):
                check_step_inputs(plan, 0, wrong)

    def test_ordinary_bf16_gradients_accumulate_in_fp32(self):
        module = torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
        p = module.weight
        p.main_grad = torch.zeros_like(p, dtype=torch.float32)
        for value in (1.0, .002, -1.0):
            p.grad = torch.full_like(p, value)
            collect_main_grads(module)
        self.assertAlmostEqual(float(p.main_grad), float(torch.tensor(.002, dtype=torch.bfloat16)), places=7)
        self.assertIsNone(p.grad)

    def test_mixed_lengths_preserve_manifest_order(self):
        fixture = {"samples": [{"sample_id": str(i), "images": [],
                                 "derived_visual_tokens_total": 10} for i in range(2)]}
        def build(processor, sample, root):
            return {"tokens": torch.zeros((1, 4 + int(sample["sample_id"])), dtype=torch.long)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "manifest.json")
            path.write_text(json.dumps(fixture))
            with patch("manifest_data.load_processor", return_value=object()), \
                    patch("manifest_data.build_sample", side_effect=build):
                data = ManifestBatches("unused", path, tmp, batching="manifest")
                self.assertEqual([s["sample_id"] for s in data.plan(0, 2)], ["0", "1"])
                self.assertEqual([b["tokens"].shape[1] for b in data.build(0, 2)], [4, 5])
                with self.assertRaises(AssertionError):
                    ManifestBatches("unused", path, tmp, batching="bucket").build(0, 2)

    def test_large_parameter_sample_indices(self):
        length = 151936 * 1536
        indices = sample_indices(length, 16, "cpu")
        self.assertEqual(int(indices[0]), 0)
        self.assertEqual(int(indices[-1]), length - 1)
        self.assertTrue(bool((indices < length).all()))

    def test_shared_manifest_smoke_without_derived_counts(self):
        fixture = {"samples": [{"sample_id": "photo", "images": [{"resolution": 128}]}]}
        batch = {"tokens": torch.tensor([[5, 7, 9, 2]]),
                 "labels": torch.tensor([[-100, -100, 9, 2]]),
                 "image_mask": torch.tensor([[False, True, False, False]]),
                 "grid": torch.tensor([[1, 2, 2]])}
        seen_targets = []
        def masks(tokens, grid, video, targets, *args, **kwargs):
            seen_targets.append(targets.clone())
            return None, targets != -100, torch.zeros((3, 1, 4), dtype=torch.long)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "manifest.json")
            path.write_text(json.dumps(fixture))
            with patch("manifest_data.load_processor", return_value=object()), \
                    patch("manifest_data.build_sample", return_value=batch) as build:
                source = ManifestBatches("unused", path, tmp)
                prepared = source.step_tensors(0, 2, "cpu", masks, None)
                self.assertEqual(build.call_count, 1)  # Cached across repeated microbatches.
                self.assertEqual(len(prepared), 2)
                torch.testing.assert_close(seen_targets[0], torch.tensor([[-100, 9, 2, -100]]))

    def test_varied_worlds_and_unequal_work(self):
        for world in (2, 3, 4):
            for count in (1, 5, 8):
                best = choose_plan(world, count, costs(world, count))
                validate_plan(best)
                serial = make_plan(world, [0] * count, costs(world, count), "serial")
                self.assertLessEqual(best["estimated_time_ms"], serial["estimated_time_ms"])

    def test_corrupt_plans_rejected(self):
        plan = make_plan(2, [0, 1], costs(2, 2))
        bad = deepcopy(plan)
        bad["events"].pop()
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_plan(bad)
        bad = deepcopy(plan)
        event = next(e for e in bad["events"] if e["kind"] == "EB")
        event.update(start=0.0, end=.1)
        with self.assertRaisesRegex(ValueError, "dependency"):
            validate_plan(bad)
        with self.assertRaises(ValueError):
            make_plan(2, [2], costs(2, 1))

    def test_two_process_training_matches_unpartitioned_reference(self):
        for world in (2, 4):
            with self.subTest(world=world), tempfile.TemporaryDirectory(prefix="optimus-test-") as tmp:
                mp.spawn(worker, args=(str(Path(tmp, "rendezvous")), tmp, world),
                         nprocs=world, join=True)
                for rank in range(world):
                    result = json.loads(Path(tmp, "rank%d.json" % rank).read_text())
                    self.assertEqual(result["optimizer_updates"], 3)
                    self.assertLess(result["max_parameter_error"], 1e-6)
                print("world=%d: three updates match unpartitioned reference" % world)

    def test_fused_gradient_is_not_counted_twice(self):
        module = torch.nn.Linear(2, 2, bias=False)
        p = module.weight
        p.main_grad = torch.full_like(p, 3.)
        p.grad = torch.full_like(p, 99.)  # TE's dummy gradient must be ignored
        p.grad_added_to_main_grad = True
        collect_main_grads(module)
        torch.testing.assert_close(p.main_grad, torch.full_like(p, 3.))
        zero_grads(module)
        self.assertFalse(p.grad_added_to_main_grad)


if __name__ == "__main__":
    unittest.main()

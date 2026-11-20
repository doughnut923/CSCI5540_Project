import unittest

import torch

from mps_compat import FlattenMPSScatterSource


class ScatterGradientContract(unittest.TestCase):
    def check_device(self, device, dtype):
        source = torch.arange(12, device=device, dtype=dtype).reshape(3, 4).requires_grad_()
        destination = torch.zeros((1, 3, 4), device=device, dtype=dtype)
        with FlattenMPSScatterSource() as compatibility:
            output = destination.masked_scatter(torch.ones_like(destination, dtype=torch.bool), source)
        output.float().square().sum().backward()
        self.assertEqual(tuple(source.shape), (3, 4))
        self.assertEqual(tuple(source.grad.shape), (3, 4))
        torch.testing.assert_close(source.grad.float().cpu(), 2 * source.detach().float().cpu())
        self.assertEqual(compatibility.applied_calls, int(device == "mps"))

    def test_cpu_is_unchanged(self):
        self.check_device("cpu", torch.float32)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_mps_float32_and_bfloat16(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self.check_device("mps", dtype)


if __name__ == "__main__":
    unittest.main()

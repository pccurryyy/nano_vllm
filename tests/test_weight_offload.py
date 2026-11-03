import copy
import unittest
import torch
from torch import nn

from nanovllm.engine.weight_offload import WeightOffloadManager


class WeightOffloadTest(unittest.TestCase):
    device = "cpu"

    def test_window_outputs_and_cpu_master_preservation(self):
        torch.manual_seed(87)
        layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])
        original = copy.deepcopy(layers).to(self.device)
        manager = WeightOffloadManager(layers, 3, window_size=2, device=self.device)
        masters = [[x.clone() for _, x in manager.masters[i]] for i in sorted(manager.masters)]
        for _ in range(2):
            x = torch.randn(3, 8, device=self.device)
            expected = x
            for layer in original:
                expected = layer(expected)
            actual = x
            manager.begin_forward()
            try:
                for i in range(4):
                    actual = manager.prepare_layer(i)(actual)
                    self.assertLessEqual(len(manager.active), 2)
            finally:
                manager.end_forward()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertFalse(manager.active)
            for i in manager.masters:
                self.assertTrue(all(p.device.type == "cpu" for p, _ in manager.masters[i]))
        for saved, i in zip(masters, sorted(manager.masters)):
            for value, (_, master) in zip(saved, manager.masters[i]):
                torch.testing.assert_close(value, master, rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class GPUWeightOffloadTest(WeightOffloadTest):
    device = "cuda"

    def test_cpu_constructed_rope_rebuilt_on_execution_device_matches_baseline(self):
        from nanovllm.layers.rotary_embedding import RotaryEmbedding
        with torch.device("cpu"):
            offloaded = RotaryEmbedding(128, 128, 512, 1000000.)
        offloaded.rebuild_cache("cuda")
        with torch.device("cuda"):
            resident = RotaryEmbedding(128, 128, 512, 1000000.)
        torch.testing.assert_close(offloaded.cos_sin_cache, resident.cos_sin_cache, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()

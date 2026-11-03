import unittest
import torch

from nanovllm.layers.attention import Attention
from nanovllm.utils.context import reset_context, set_context


class TorchAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.attn = Attention(4, 8, 8 ** -0.5, 2)
        self.attn.use_flash = False
        self.attn.k_cache = torch.zeros(3, 4, 2, 8)
        self.attn.v_cache = torch.zeros_like(self.attn.k_cache)

    def tearDown(self):
        reset_context()

    def reference(self, q, k, v):
        k = k.repeat_interleave(2, dim=1)
        v = v.repeat_interleave(2, dim=1)
        scores = torch.einsum('qhd,khd->hqk', q, k) * self.attn.scale
        mask = torch.arange(k.size(0))[None, :] > torch.arange(k.size(0) - q.size(0), k.size(0))[:, None]
        weights = scores.masked_fill(mask, -float('inf')).softmax(-1)
        return torch.einsum('hqk,khd->qhd', weights, v)

    def test_prefill_decode_and_cached_chunk(self):
        q, k, v = torch.randn(7, 4, 8), torch.randn(7, 2, 8), torch.randn(7, 2, 8)
        # Two prompts of different lengths, with non-contiguous physical pages.
        set_context(True, cu_seqlens_q=torch.tensor([0, 3, 5]),
                    cu_seqlens_k=torch.tensor([0, 3, 5]),
                    slot_mapping=torch.tensor([8, 9, 10, 0, 1]))
        actual = self.attn(q[:5], k[:5], v[:5])
        expected = torch.cat([self.reference(q[:3], k[:3], v[:3]),
                              self.reference(q[3:5], k[3:5], v[3:5])])
        torch.testing.assert_close(actual, expected)
        set_context(False, slot_mapping=torch.tensor([11, 2]),
                    context_lens=torch.tensor([4, 3]),
                    block_tables=torch.tensor([[2], [0]]))
        actual = self.attn(q[5:7], k[5:7], v[5:7])
        torch.testing.assert_close(actual[0:1], self.reference(q[5:6], torch.cat([k[:3], k[5:6]]), torch.cat([v[:3], v[5:6]])))
        torch.testing.assert_close(actual[1:2], self.reference(q[6:7], torch.cat([k[3:5], k[6:7]]), torch.cat([v[3:5], v[6:7]])))
        # A two-token prefill chunk attending to a four-token cached prefix.
        set_context(True, cu_seqlens_q=torch.tensor([0, 2]),
                    cu_seqlens_k=torch.tensor([0, 6]),
                    slot_mapping=torch.tensor([4, 5]),
                    block_tables=torch.tensor([[2, 1]]))
        actual = self.attn(q[:2], k[:2], v[:2])
        prefix_k, prefix_v = torch.cat([k[:3], k[5:6]]), torch.cat([v[:3], v[5:6]])
        torch.testing.assert_close(actual, self.reference(q[:2], torch.cat([prefix_k, k[:2]]), torch.cat([prefix_v, v[:2]])))


if __name__ == '__main__':
    unittest.main()

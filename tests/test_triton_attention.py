import unittest
import torch

import test_torch_attention as reference_tests
from nanovllm.layers.attention import store_kvcache
from nanovllm.utils.context import set_context


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TritonAttentionTest(reference_tests.TorchAttentionTest):
    def setUp(self):
        super().setUp()
        self.attn.backend = "triton"
        self.attn.k_cache = self.attn.k_cache.cuda()
        self.attn.v_cache = self.attn.v_cache.cuda()
        # The parent test creates inputs and context tensors on this device.
        torch.set_default_device("cuda")

    def tearDown(self):
        super().tearDown()
        torch.set_default_device("cpu")

    def test_original_store_kernel(self):
        k = torch.randn(4, 8, 128, dtype=torch.float16)
        v = torch.randn_like(k)
        kc = torch.zeros(2, 256, 8, 128, dtype=torch.float16)
        vc = torch.zeros_like(kc)
        slots = torch.tensor([0, 257, -1, 511], dtype=torch.int32)
        store_kvcache(k, v, kc, vc, slots)
        for i, slot in [(0, 0), (1, 257), (3, 511)]:
            torch.testing.assert_close(kc.flatten(0, 1)[slot], k[i])
            torch.testing.assert_close(vc.flatten(0, 1)[slot], v[i])
        self.assertEqual(kc.flatten(0, 1)[1:257].count_nonzero().item(), 0)

    def test_fp16_ragged_prefill_with_strided_qkv(self):
        self.attn.num_heads, self.attn.num_kv_heads = 16, 8
        self.attn.head_dim, self.attn.scale = 128, 128 ** -0.5
        self.attn.k_cache = self.attn.v_cache = torch.tensor([])
        packed = torch.randn(350, 4096, dtype=torch.float16)
        q, k, v = packed.split([2048, 1024, 1024], -1)
        q, k, v = q.view(350, 16, 128), k.view(350, 8, 128), v.view(350, 8, 128)
        offsets = torch.tensor([0, 33, 350], dtype=torch.int32)
        set_context(True, cu_seqlens_q=offsets, cu_seqlens_k=offsets)
        actual = self.attn(q, k, v)
        reference = []
        for start, end in [(0, 33), (33, 350)]:
            reference.append(torch.nn.functional.scaled_dot_product_attention(
                q[start:end].transpose(0, 1).float(),
                k[start:end].repeat_interleave(2, 1).transpose(0, 1).float(),
                v[start:end].repeat_interleave(2, 1).transpose(0, 1).float(),
                is_causal=True, scale=self.attn.scale,
            ).transpose(0, 1).half())
        torch.testing.assert_close(actual, torch.cat(reference), atol=1e-3, rtol=1e-3)

    def test_graph_replay_with_changing_pages_and_lengths(self):
        # Match Qwen3's actual head dimension and GQA ratio.
        self.attn.num_heads, self.attn.num_kv_heads = 16, 8
        self.attn.head_dim, self.attn.scale = 128, 128 ** -0.5
        self.attn.k_cache = torch.randn(3, 256, 8, 128, dtype=torch.float16)
        self.attn.v_cache = torch.randn_like(self.attn.k_cache)
        q = torch.randn(2, 16, 128, dtype=torch.float16)
        k, v = torch.randn(2, 8, 128, dtype=torch.float16), torch.randn(2, 8, 128, dtype=torch.float16)
        lengths = torch.tensor([33, 0], dtype=torch.int32)
        pages = torch.tensor([[2, 0], [1, 0]], dtype=torch.int32)
        slots = torch.tensor([544, -1], dtype=torch.int32)
        set_context(False, slot_mapping=slots, context_lens=lengths, block_tables=pages)
        self.attn(q, k, v)  # Compile before capture.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = self.attn(q, k, v)
        for length in (34, 260):
            lengths[0] = length
            slots[0] = 2 * 256 + length - 1 if length <= 256 else length - 257
            graph.replay()
            torch.cuda.synchronize()
            key = self.attn.k_cache[pages[0].long()].flatten(0, 1)[:length]
            value = self.attn.v_cache[pages[0].long()].flatten(0, 1)[:length]
            reference = torch.nn.functional.scaled_dot_product_attention(
                q[0].unsqueeze(1).float(),
                key.repeat_interleave(2, 1).transpose(0, 1).float(),
                value.repeat_interleave(2, 1).transpose(0, 1).float(), scale=self.attn.scale,
            ).transpose(0, 1).half()
            torch.testing.assert_close(captured[:1], reference, atol=1e-3, rtol=1e-3)
            self.assertEqual(captured[1].count_nonzero().item(), 0)

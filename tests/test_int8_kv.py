import unittest
import torch

from nanovllm.layers.attention import Attention
from nanovllm.layers.kv_cache import store_int8, quantize_reference, cache_block_bytes, INT8_CACHE, MIXED_CACHE
from nanovllm.utils.context import set_context, reset_context, get_context


class QuantizationSpecTest(unittest.TestCase):
    def test_rounding_zero_outliers_and_size(self):
        x = torch.tensor([[[127., .5, -.5, 1.5, -1.5, 0., -127., 2.5]]])
        q, s = quantize_reference(x)
        self.assertEqual(q.tolist(), [[[127, 1, -1, 2, -2, 0, -127, 3]]])
        self.assertEqual(s.item(), 1.)
        q, s = quantize_reference(torch.zeros(2, 2, 128))
        self.assertEqual(q.count_nonzero().item(), 0)
        self.assertTrue(torch.all(s == 1e-6))
        self.assertEqual(cache_block_bytes(28, 256, 8, 128, torch.float16, True), int(14.4375 * 2**20))
        for bad in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                quantize_reference(torch.tensor([[[bad]]]))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class Int8KernelTest(unittest.TestCase):
    mode = INT8_CACHE

    def setUp(self):
        torch.manual_seed(71)
        self.a = Attention(4, 128, 128 ** -.5, 2)
        self.a.backend = "triton"
        self.a.kv_cache_dtype = self.mode
        self.a.k_cache = torch.zeros(3, 256, 2, 128, dtype=torch.int8 if self.mode == INT8_CACHE else torch.float16, device="cuda")
        self.a.v_cache = torch.zeros_like(self.a.k_cache, dtype=torch.int8)
        self.a.k_scale_cache = torch.ones(3, 256, 2, device="cuda") if self.mode == INT8_CACHE else None
        self.a.v_scale_cache = torch.ones(3, 256, 2, device="cuda")

    def store(self, *args):
        store_int8(*args, quantize_key=self.mode == INT8_CACHE)

    def tearDown(self):
        reset_context()

    def test_strided_write_and_invalid_slots(self):
        packed = torch.randn(4, 2, 384, device="cuda", dtype=torch.float16)
        k, v = packed[..., :128], packed[..., 128:256]
        k[0, 0, :8] = torch.tensor([127, .5, -.5, 1.5, -1.5, 0, -127, 2.5], device="cuda")
        k[1].zero_()
        k[2, 0, 7] = 1000
        slots = torch.tensor([0, 257, -1, 767], device="cuda", dtype=torch.int32)
        self.store(k, v, self.a.k_cache, self.a.v_cache,
                   self.a.k_scale_cache, self.a.v_scale_cache, slots)
        for data, cache, scales in [(k, self.a.k_cache, self.a.k_scale_cache),
                                    (v, self.a.v_cache, self.a.v_scale_cache)]:
            expected, scale = quantize_reference(data) if scales is not None else (data, None)
            for i, slot in [(0, 0), (1, 257), (3, 767)]:
                torch.testing.assert_close(cache.flatten(0, 1)[slot], expected[i], rtol=0, atol=0)
                if scales is not None:
                    torch.testing.assert_close(scales.flatten(0, 1)[slot], scale[i], rtol=1e-6, atol=1e-8)
            self.assertEqual(cache.flatten(0, 1)[2].count_nonzero().item(), 0)
            if scales is not None:
                self.assertTrue(torch.all(scales.flatten(0, 1)[2] == 1))

    def test_fp16_half_boundary_bulk_write(self):
        # FP16 normalization can land within an ulp of a half integer.
        k = torch.randn(270, 2, 128, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
        slots = torch.arange(270, device="cuda", dtype=torch.int32)
        self.store(k, v, self.a.k_cache, self.a.v_cache,
                   self.a.k_scale_cache, self.a.v_scale_cache, slots)
        for x, c in [(k, self.a.k_cache), (v, self.a.v_cache)]:
            expected = quantize_reference(x)[0] if c.dtype == torch.int8 else x
            torch.testing.assert_close(c.flatten(0, 1)[:270], expected, rtol=0, atol=0)

    def test_noncontiguous_scale_views(self):
        backing_k = torch.ones(3, 256, 2, 2, device="cuda")
        backing_v = torch.ones_like(backing_k)
        self.a.k_scale_cache = backing_k[..., 0] if self.mode == INT8_CACHE else None
        self.a.v_scale_cache = backing_v[..., 0]
        q = torch.randn(3, 4, 128, device="cuda", dtype=torch.float16)
        k = torch.randn(3, 2, 128, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
        set_context(True, cu_seqlens_q=torch.tensor([0, 3], device="cuda", dtype=torch.int32),
                    cu_seqlens_k=torch.tensor([0, 3], device="cuda", dtype=torch.int32),
                    slot_mapping=torch.tensor([512, 513, 514], device="cuda", dtype=torch.int32),
                    block_tables=torch.tensor([[2]], device="cuda", dtype=torch.int32))
        self.compare(q, k, v)
        self.assertTrue(torch.all(backing_k[..., 1] == 1))
        self.assertTrue(torch.all(backing_v[..., 1] == 1))

    def compare(self, q, k, v):
        actual = self.a(q, k, v)
        self.a.backend = "torch"
        expected = self.a.torch_forward(q, k, v, get_context(), already_stored=True)
        self.a.backend = "triton"
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
        return actual

    def test_prefill_decode_and_graph(self):
        q = torch.randn(270, 4, 128, device="cuda", dtype=torch.float16)
        k = torch.randn(270, 2, 128, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
        slots = torch.cat([torch.arange(512, 768, device="cuda"), torch.arange(14, device="cuda")]).int()
        pages = torch.tensor([[2, 0]], device="cuda", dtype=torch.int32)
        set_context(True, cu_seqlens_q=torch.tensor([0, 270], device="cuda", dtype=torch.int32),
                    cu_seqlens_k=torch.tensor([0, 270], device="cuda", dtype=torch.int32),
                    slot_mapping=slots, block_tables=pages)
        self.compare(q, k, v)
        # Append one token, plus a graph padding row. Data and scale must not
        # be touched by the padding row; zero-context output must be zero.
        lens = torch.tensor([271, 0], device="cuda", dtype=torch.int32)
        slots = torch.tensor([14, -1], device="cuda", dtype=torch.int32)
        pages = torch.tensor([[2, 0], [1, 0]], device="cuda", dtype=torch.int32)
        set_context(False, context_lens=lens, block_tables=pages, slot_mapping=slots)
        out = self.a(q[:2], k[:2], v[:2])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = self.a(q[:2], k[:2], v[:2])
        for length in (272, 33, 260):
            lens[0] = length
            slots[0] = 512 + length - 1 if length <= 256 else length - 257
            graph.replay()
            torch.cuda.synchronize()
            self.a.backend = "torch"
            # Reference sees only a real request; its Python path is not used for padding.
            set_context(False, context_lens=lens[:1], block_tables=pages[:1], slot_mapping=slots[:1])
            ref = self.a(q[:1], k[:1], v[:1])
            self.a.backend = "triton"
            set_context(False, context_lens=lens, block_tables=pages, slot_mapping=slots)
            torch.testing.assert_close(out[:1], ref, rtol=1e-3, atol=1e-3)
            self.assertEqual(out[1].count_nonzero().item(), 0)
            self.assertEqual(self.a.k_cache[1].count_nonzero().item(), 0)
            if self.a.k_scale_cache is not None:
                self.assertTrue(torch.all(self.a.k_scale_cache[1] == 1))

    def test_chunk_and_prefix_are_equivalent(self):
        q = torch.randn(270, 4, 128, device="cuda", dtype=torch.float16)
        k = torch.randn(270, 2, 128, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
        slots = torch.arange(270, device="cuda", dtype=torch.int32)
        pages = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
        def run(start, end):
            set_context(True, cu_seqlens_q=torch.tensor([0, end-start], device="cuda", dtype=torch.int32),
                        cu_seqlens_k=torch.tensor([0, end], device="cuda", dtype=torch.int32),
                        slot_mapping=slots[start:end], block_tables=pages)
            return self.a(q[start:end], k[start:end], v[start:end])
        full = run(0, 270)
        chunked = torch.cat([run(0, 128), run(128, 256), run(256, 270)])
        torch.testing.assert_close(full, chunked, rtol=0, atol=0)
        reused = run(256, 270)
        torch.testing.assert_close(full[256:], reused, rtol=0, atol=0)


class MixedKernelTest(Int8KernelTest):
    mode = MIXED_CACHE


if __name__ == "__main__":
    unittest.main()

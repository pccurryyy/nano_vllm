import unittest
import torch

from nanovllm.layers.attention import Attention
from nanovllm.layers.kv_cache import MIXED_CACHE, store_int8
from nanovllm.utils.context import set_context, get_context, reset_context


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class MixedAttentionTest(unittest.TestCase):
    def tearDown(self):
        reset_context()

    def test_decode_and_prefill_against_same_cache_sdpa(self):
        # Request A: cached 258 + decode 1. B: cached 254 + prefill 5.
        # Both cross page boundaries; physical order differs from logical order.
        for fmt in ("auto", MIXED_CACHE):
            with self.subTest(fmt=fmt):
                torch.manual_seed(572)
                a = Attention(4, 128, 128 ** -.5, 2)
                a.backend = "triton"
                a.kv_cache_dtype = fmt
                a.k_cache = torch.randn(4, 256, 2, 128, device="cuda", dtype=torch.float16)
                a.v_cache = torch.randn_like(a.k_cache)
                if fmt == MIXED_CACHE:
                    k, v = a.k_cache.flatten(0, 1), a.v_cache.flatten(0, 1)
                    a.v_cache = torch.zeros_like(a.k_cache, dtype=torch.int8)
                    a.v_scale_cache = torch.empty(4, 256, 2, device="cuda", dtype=torch.float32)
                    store_int8(k, v, a.k_cache, a.v_cache, None, a.v_scale_cache,
                               torch.arange(1024, device="cuda", dtype=torch.int32), quantize_key=False)
                q = torch.randn(6, 4, 128, device="cuda", dtype=torch.float16)
                k = torch.randn(6, 2, 128, device="cuda", dtype=torch.float16)
                v = torch.randn_like(k)
                table = torch.tensor([[2, 0], [3, 1]], device="cuda", dtype=torch.int32)
                slots = torch.tensor([2, 1022, 1023, 256, 257, 258], device="cuda", dtype=torch.int32)
                set_context(True,
                    cu_seqlens_q=torch.tensor([0, 1, 6], device="cuda", dtype=torch.int32),
                    cu_seqlens_k=torch.tensor([0, 259, 518], device="cuda", dtype=torch.int32),
                    slot_mapping=slots, block_tables=table)
                actual = a(q, k, v)
                expected = a.torch_forward(q, k, v, get_context(), already_stored=True)
                torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()

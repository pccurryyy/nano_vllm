import torch
from torch import nn
import torch.nn.functional as F
import triton
import triton.language as tl

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except ImportError:
    flash_attn_varlen_func = flash_attn_with_kvcache = None
from nanovllm.utils.context import get_context
from nanovllm.layers.triton_attention import triton_attention
from nanovllm.layers.kv_cache import store_int8, INT8_CACHE, MIXED_CACHE


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.use_flash = (flash_attn_varlen_func is not None
                          and torch.cuda.is_available()
                          and torch.cuda.get_device_capability()[0] >= 8)
        self.backend = "flash" if self.use_flash else "torch"
        self.k_cache = self.v_cache = torch.tensor([])
        self.k_scale_cache = self.v_scale_cache = None
        self.kv_cache_dtype = "auto"

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        quant = self.kv_cache_dtype in (INT8_CACHE, MIXED_CACHE) and k_cache.numel() > 0
        if quant:
            if context.block_tables is None:
                raise ValueError("Quantized cache must be read through block_tables")
            store_int8(k, v, k_cache, v_cache, self.k_scale_cache,
                       self.v_scale_cache, context.slot_mapping,
                       reference=self.backend == "torch",
                       quantize_key=self.kv_cache_dtype == INT8_CACHE)
            if self.backend == "triton":
                return triton_attention(q, k_cache, v_cache, context, self.scale,
                                        k_cache.size(1), self.k_scale_cache, self.v_scale_cache)
            return self.torch_forward(q, k, v, context, already_stored=True)
        if self.backend == "triton":
            if k_cache.numel():
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            if context.block_tables is not None:
                k, v = k_cache, v_cache
            return triton_attention(q, k, v, context, self.scale,
                                    k_cache.size(1) if k_cache.numel() else 256)
        if not self.use_flash:
            return self.torch_forward(q, k, v, context)
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o

    def torch_forward(self, q, k, v, context, already_stored=False):
        """SDPA fallback with the same paged KV cache and causal alignment."""
        if self.k_cache.numel() and not already_stored:
            slots = context.slot_mapping.long()
            valid = slots >= 0
            self.k_cache.flatten(0, 1).index_copy_(0, slots[valid], k[valid])
            self.v_cache.flatten(0, 1).index_copy_(0, slots[valid], v[valid])

        if context.is_prefill:
            q_offsets = context.cu_seqlens_q.tolist()
            k_offsets = context.cu_seqlens_k.tolist()
            lengths = [b - a for a, b in zip(k_offsets, k_offsets[1:])]
        else:
            q_offsets = list(range(q.size(0) + 1))
            lengths = context.context_lens.tolist()

        outputs = []
        for i, length in enumerate(lengths):
            query = q[q_offsets[i]:q_offsets[i + 1]]
            if context.block_tables is not None:
                block_size = self.k_cache.size(1)
                blocks = context.block_tables[i, :(length + block_size - 1) // block_size].long()
                key = self.k_cache[blocks].flatten(0, 1)[:length]
                value = self.v_cache[blocks].flatten(0, 1)[:length]
                if self.kv_cache_dtype in (INT8_CACHE, MIXED_CACHE):
                    if self.k_scale_cache is not None:
                        ks = self.k_scale_cache[blocks].flatten(0, 1)[:length]
                        key = key.float() * ks.unsqueeze(-1)
                    vs = self.v_scale_cache[blocks].flatten(0, 1)[:length]
                    value = value.float() * vs.unsqueeze(-1)
            else:
                key = k[k_offsets[i]:k_offsets[i + 1]]
                value = v[k_offsets[i]:k_offsets[i + 1]]
            repeats = self.num_heads // self.num_kv_heads
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)
            query_len = query.size(0)
            mask = (torch.arange(length, device=q.device)[None, :]
                    <= torch.arange(length - query_len, length, device=q.device)[:, None])
            out = F.scaled_dot_product_attention(
                query.transpose(0, 1).unsqueeze(0).float(),
                key.transpose(0, 1).unsqueeze(0).float(),
                value.transpose(0, 1).unsqueeze(0).float(),
                attn_mask=mask, scale=self.scale,
            )
            outputs.append(out.squeeze(0).transpose(0, 1).to(q.dtype))
        return torch.cat(outputs)

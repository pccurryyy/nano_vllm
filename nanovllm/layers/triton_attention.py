"""Streaming attention using CUDA cores (no Tensor Core instructions).

This implements tiled online softmax, not the upstream flash-attn library.
Both ragged prefill and paged decode stay on device for CUDA Graph capture.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    Q, K, V, O, CU_Q, CU_K, LENS, TABLE, KS, VS,
    TABLE_STRIDE: tl.constexpr, NUM_SEQS: tl.constexpr,
    H: tl.constexpr, KV_H: tl.constexpr, D: tl.constexpr,
    PAGE: tl.constexpr, SCALE: tl.constexpr,
    PREFILL: tl.constexpr, PAGED: tl.constexpr,
    SEQ_PAD: tl.constexpr, TILE: tl.constexpr,
    QUANT: tl.constexpr, QUANT_K: tl.constexpr,
    KSB: tl.constexpr, KSP: tl.constexpr, KSH: tl.constexpr,
    VSB: tl.constexpr, VSP: tl.constexpr, VSH: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    kv_head = head // (H // KV_H)
    dims = tl.arange(0, D)
    query = tl.load(Q + token * H * D + head * D + dims).to(tl.float32)
    if PREFILL:
        ids = tl.arange(0, SEQ_PAD)
        ends = tl.load(CU_Q + ids + 1, ids < NUM_SEQS, other=2147483647)
        seq = tl.sum((token >= ends).to(tl.int32), 0)
        q_start = tl.load(CU_Q + seq)
        q_end = tl.load(CU_Q + seq + 1)
        k_start = tl.load(CU_K + seq)
        k_end = tl.load(CU_K + seq + 1)
        length = k_end - k_start - (q_end - q_start) + token - q_start + 1
    else:
        seq = token
        length = tl.load(LENS + seq)
        k_start = 0
    maximum = tl.full((), -float('inf'), tl.float32)
    denominator = tl.full((), 0, tl.float32)
    accumulator = tl.full((D,), 0, tl.float32)
    for tile in range(tl.cdiv(length, TILE)):
        positions = tile * TILE + tl.arange(0, TILE)
        valid = positions < length
        if PAGED:
            pages = tl.load(TABLE + seq * TABLE_STRIDE + positions // PAGE,
                            valid, other=0)
            slots = pages * PAGE + positions % PAGE
        else:
            slots = k_start + positions
        offsets = slots[:, None] * KV_H * D + kv_head * D + dims[None, :]
        key = tl.load(K + offsets, valid[:, None], other=0).to(tl.float32)
        value = tl.load(V + offsets, valid[:, None], other=0).to(tl.float32)
        if QUANT_K:
            ks = tl.load(KS + pages * KSB + (positions % PAGE) * KSP + kv_head * KSH,
                         valid, other=1.)
            key *= ks[:, None]
        if QUANT:
            vs = tl.load(VS + pages * VSB + (positions % PAGE) * VSP + kv_head * VSH,
                         valid, other=1.)
            value *= vs[:, None]
        scores = tl.sum(key * query[None, :], 1) * SCALE
        scores = tl.where(valid, scores, -float('inf'))
        new_max = tl.maximum(maximum, tl.max(scores, 0))
        alpha = tl.exp(maximum - new_max)
        probabilities = tl.exp(scores - new_max)
        accumulator = accumulator * alpha + tl.sum(probabilities[:, None] * value, 0)
        denominator = denominator * alpha + tl.sum(probabilities, 0)
        maximum = new_max
    result = tl.where(denominator > 0, accumulator / denominator, 0)
    tl.store(O + token * H * D + head * D + dims, result)


def triton_attention(q, k, v, context, scale, block_size, k_scales=None, v_scales=None):
    q = q.contiguous()
    quant = v_scales is not None
    if quant:
        # Never silently materialize a full cache copy in the compressed path.
        if not k.is_contiguous() or not v.is_contiguous():
            raise ValueError("INT8 cache data must have the allocated contiguous layout")
    else:
        k, v = k.contiguous(), v.contiguous()
    out = torch.empty_like(q)
    paged = context.block_tables is not None
    if quant and not paged:
        raise ValueError("INT8 attention requires paged context")
    num_seqs = context.cu_seqlens_q.numel() - 1 if context.is_prefill else q.size(0)
    attention_kernel[(q.size(0), q.size(1))](
        q, k, v, out, context.cu_seqlens_q, context.cu_seqlens_k,
        context.context_lens, context.block_tables, k_scales, v_scales,
        context.block_tables.stride(0) if paged else 0, num_seqs,
        q.size(1), k.size(-2), q.size(-1), block_size, scale,
        context.is_prefill, paged, triton.next_power_of_2(num_seqs), 32,
        quant, k_scales is not None, *(k_scales.stride() if k_scales is not None else (0, 0, 0)),
        *(v_scales.stride() if quant else (0, 0, 0)),
        num_warps=4,
    )
    return out

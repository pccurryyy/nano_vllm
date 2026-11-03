# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged INT8 KV storage, following vLLM's per-token-head quantization.

Reference: vllm-project/vllm commit 23fbc2ec2f7219888bfc76cf987381f501151971,
vllm/v1/attention/ops/triton_reshape_and_cache_flash.py (Apache-2.0).
This independently integrated kernel retains absmax scaling, FP32 scales,
half-away-from-zero rounding and negative-slot handling. Attention arithmetic
is provided by nano-vLLM's SM75 streaming kernel, not vLLM's GEMM kernels.
"""
import torch
import triton
import triton.language as tl


INT8_CACHE = "int8_per_token_head"
MIXED_CACHE = "fp16_k_int8_v"


def cache_block_bytes(layers, page, heads, dim, dtype, quantized=False, value_only=False):
    if value_only:
        return layers * page * heads * (dim * dtype.itemsize + dim + 4)
    return 2 * layers * page * heads * (dim + 4 if quantized else dim * dtype.itemsize)


def quantize_reference(x):
    """Executable rounding specification; also used by the eager reference."""
    x = x.float()
    if not torch.isfinite(x).all().item():
        raise ValueError("KV quantization requires finite inputs")
    scales = (x.abs().amax(-1) / 127).clamp_min(1e-6)
    z = x / scales.unsqueeze(-1)
    rounded = torch.sign(z) * torch.floor(z.abs() + 0.5)
    return rounded.clamp(-128, 127).to(torch.int8), scales


@triton.jit
def _store_int8(K, V, KC, VC, KS, VS, SLOTS,
                KT: tl.constexpr, KH: tl.constexpr, VT: tl.constexpr, VH: tl.constexpr,
                CB: tl.constexpr, CP: tl.constexpr, CH: tl.constexpr,
                VB: tl.constexpr, VP: tl.constexpr, VCH: tl.constexpr,
                SB: tl.constexpr, SP: tl.constexpr, SH: tl.constexpr,
                VSB: tl.constexpr, VSP: tl.constexpr, VSH: tl.constexpr,
                PAGE: tl.constexpr, D: tl.constexpr, PAD: tl.constexpr, QUANT_K: tl.constexpr):
    token, head = tl.program_id(0), tl.program_id(1)
    slot = tl.load(SLOTS + token).to(tl.int64)
    if slot < 0:
        return
    block, offset = slot // PAGE, slot % PAGE
    dims = tl.arange(0, PAD)
    k = tl.load(K + token * KT + head * KH + dims, dims < D, other=0).to(tl.float32)
    v = tl.load(V + token * VT + head * VH + dims, dims < D, other=0).to(tl.float32)
    if QUANT_K:
        ks = tl.maximum(tl.max(tl.abs(k), 0) / 127., 1e-6)
        kz = tl.div_rn(k, ks)
        kz = tl.where(kz >= 0, kz + 0.5, kz - 0.5)
        tl.store(KC + block * CB + offset * CP + head * CH + dims,
                 tl.minimum(tl.maximum(kz, -128.), 127.).to(tl.int8), dims < D)
        tl.store(KS + block * SB + offset * SP + head * SH, ks)
    else:
        tl.store(KC + block * CB + offset * CP + head * CH + dims, k, dims < D)
    vs = tl.maximum(tl.max(tl.abs(v), 0) / 127., 1e-6)
    # Exact division and half-away-from-zero rounding match the specification.
    vz = tl.div_rn(v, vs)
    vz = tl.where(vz >= 0, vz + 0.5, vz - 0.5)
    tl.store(VC + block * VB + offset * VP + head * VCH + dims,
             tl.minimum(tl.maximum(vz, -128.), 127.).to(tl.int8), dims < D)
    tl.store(VS + block * VSB + offset * VSP + head * VSH, vs)


def store_int8(k, v, kc, vc, ks, vs, slots, reference=False, quantize_key=True):
    assert k.shape == v.shape and slots.numel() == k.shape[0]
    assert k.stride(-1) == v.stride(-1) == 1
    assert vc.dtype == torch.int8 and vs.dtype == torch.float32
    assert kc.shape == vc.shape and tuple(vs.shape) == tuple(vc.shape[:-1])
    assert tuple(k.shape[1:]) == tuple(kc.shape[-2:])
    if quantize_key:
        assert kc.dtype == torch.int8 and ks.dtype == torch.float32 and ks.shape == vs.shape
    else:
        assert kc.dtype == k.dtype == torch.float16 and ks is None
    if reference:
        valid = slots >= 0
        blocks = slots[valid].long() // kc.size(1)
        offsets = slots[valid].long() % kc.size(1)
        kq, kscale = quantize_reference(k[valid]) if quantize_key else (k[valid], None)
        vq, vscale = quantize_reference(v[valid])
        kc[blocks, offsets], vc[blocks, offsets] = kq, vq
        if quantize_key:
            ks[blocks, offsets] = kscale
        vs[blocks, offsets] = vscale
        return
    _store_int8[(k.size(0), k.size(1))](
        k, v, kc, vc, ks, vs, slots,
        *k.stride()[:2], *v.stride()[:2],
        *kc.stride()[:3], *vc.stride()[:3], *(ks.stride() if quantize_key else (0, 0, 0)), *vs.stride(),
        kc.size(1), k.size(-1), triton.next_power_of_2(k.size(-1)), quantize_key, num_warps=4,
        enable_fp_fusion=False,
    )

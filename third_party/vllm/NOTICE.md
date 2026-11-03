# vLLM INT8 KV reference

The per-token-head quantization in `nanovllm/layers/kv_cache.py` is adapted from
the mechanisms in vLLM commit `23fbc2ec2f7219888bfc76cf987381f501151971`:

- `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py`
- `vllm/v1/attention/ops/triton_unified_attention.py`

Copyright contributors to the vLLM project. The Apache-2.0 license is included
in this directory. Existing nano-vLLM code retains its original license.

Local differences include separate scale allocation, integration with the
existing SM75 CUDA-core streaming attention, FP32 tile dequantization and
accumulation, and exact FP32 division at integer rounding boundaries. This
does not import vLLM or claim compatibility with its complete GPU backend.

The mixed cache option reuses this value quantizer, while retaining FP16 keys.
Its independently configurable K/V precision policy is informed by llama.cpp
commit `6868a52cb337450658006ad2f9281c2eb63f1666`, `common/arg.cpp` lines
2440–2466 (`--cache-type-k` and `--cache-type-v`). No llama.cpp source code or
Q8_0 data structures were copied. The downloaded reference file SHA-256 was
`f0eccbc10ca1d214ca804fcc6d2763db1d691f2e3387692c8b64b619f858627b`.

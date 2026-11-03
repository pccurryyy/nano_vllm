# nano-vLLM-GPU-offload design reference

Source: https://github.com/HuangRZzzz/nano-vllm-GPU-offload

- GPU_OFFLOAD: `6864dede7f51d9cf149f11d0a6e5928994c2a03e`
- main: `c25959d705499b69c13f69b46bf74a7649192085`

The logical/physical block separation, protected working set with LRU eviction,
mixed decode/prefill scheduling, and decoder weight residency window informed
the local extensions. Implementation is independently integrated with the
existing scheduler and heterogeneous FP16 K / INT8 V / FP32 scale pools.

Local differences include consuming CPU copies on reload, a host scratch page
for a full-pool swap, retaining the original inactive-prefix allocator behavior,
and keeping immutable CPU master weights rather than copying evicted weights
back from the GPU. Synchronous KV transfers happen outside captured Decode
graphs. Weight offload explicitly disables compilation and CUDA Graphs.

The reference repository carries the MIT license reproduced in LICENSE.
Existing upstream nano-vLLM code remains under its original license. Reference
benchmark numbers are not local measurements and are not adopted as results.

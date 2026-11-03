<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

### Local setup on GTX 1650 (4GB)

The tested local path uses FP16, Triton 3.2.0, the original Triton KV-store
kernel, and a CUDA-core streaming attention kernel. It supports paged KV
caching, chunked prefill, prefix reuse, torch compilation, and CUDA Graphs.
The streaming kernel is an adaptation, not the upstream `flash-attn` library.
Current Triton officially lists SM80+; this SM75 path is tested with the pinned
local versions only.

```bash
python3 -m venv .venv
.venv/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install -r requirements-local.txt
.venv/bin/hf download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B
.venv/bin/python example.py --prompt "用中文简单介绍自己"
# Enable decode CUDA Graphs and compiled elementwise kernels:
.venv/bin/python example.py --backend triton --cuda-graph --compile --prompt "1加1等于多少？"
# Reference implementation:
.venv/bin/python example.py --backend torch --no-compile --prompt "你好"
```

`example.py` uses a 1024-token context, a 256-token prefill budget, and at most
two concurrent sequences. Thinking is disabled for short direct answers.
Use `--model`, `--prompt` (repeatable), and `--max-tokens` to change the input.

### Optional compressed KV cache (single GPU)

**Experimental:** `kv_cache_dtype="fp16_k_int8_v"` retains FP16 keys and quantizes values to INT8
with FP32 scales per token and KV head. On the frozen local Qwen3-0.6B heldout
set, PPL increased 0.020–0.026% across compiled/eager execution, and forced-choice
retrieval accuracy was 38% versus
37% for FP16. Including scale metadata, KV storage is reduced by 24.21875%.
Strict cross-backend
full-model logit tolerance still fails, and no throughput gain is established.
FP16 remains the default.

The value quantizer follows
[vLLM's pinned implementation](https://github.com/vllm-project/vllm/blob/23fbc2ec2f7219888bfc76cf987381f501151971/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py).
Independent K/V precision follows the configurable policy in
[llama.cpp](https://github.com/ggml-org/llama.cpp/blob/6868a52cb337450658006ad2f9281c2eb63f1666/common/arg.cpp#L2440-L2466);
this implementation does not use llama.cpp's Q8_0 binary format.
Dequantization is fused into attention tile loads. Q and weights keep their
original precision. FlashAttention, tensor parallelism, and execution dtypes
other than FP16 are unsupported for this mixed mode.

```bash
.venv/bin/python example.py --backend triton --cuda-graph --compile --kv-cache-dtype fp16_k_int8_v --max-tokens 32 --prompt "你好"
.venv/bin/python scripts/reproduce_mixed.py
```

A 256-position page across all Qwen3-0.6B layers occupies 28 MiB in FP16,
21.21875 MiB in mixed mode, or 14.4375 MiB with both K/V in INT8, including
scales. These are KV storage sizes, not total process memory.
The original `int8_per_token_head` mode remains **experimental**: it failed its
quality gates.

Requests now require a nonempty prompt, positive output limit, and
`prompt_length + max_tokens <= max_model_len`; this also protects fixed CUDA
Graph block-table capacity.

### Experimental mixed batching and CPU offload (single GPU)

The local extensions now include opt-in Decode-first mixed batches, synchronous
KV paging to CPU, and an inference-only decoder weight residency window.
KV paging migrates K, V and FP32 scales together and can be combined with the
compressed format. Its active batch's full KV working set must still fit on GPU;
this does not stream an arbitrarily long individual context from CPU.

```bash
.venv/bin/python example.py --backend triton --no-compile --stable-fp16 \
  --mixed-batch --cpu-kv-blocks 4 --kv-blocks 4 \
  --kv-cache-dtype fp16_k_int8_v --max-tokens 8 --prompt "你好"
```

Use `--weight-offload-layers N --weight-window 1` to keep the last N decoder
layers' master weights on CPU. Weight offload forces eager, uncompiled execution;
KV offload alone can keep pure Decode CUDA Graphs, while mixed batches run via
the variable-length eager path. `--stable-fp16` disables reduced-precision FP16
GEMM reductions; it is a process-wide numerical setting, not a speed guarantee.
These paths remain experimental: batching/recomputation changes can still exceed
strict full-model logit tolerance with the lossy mixed cache, even when
same-schedule migration checks pass. Defaults and the original baseline remain
available. See [source attribution](third_party/nano_vllm_offload/NOTICE.md).

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)

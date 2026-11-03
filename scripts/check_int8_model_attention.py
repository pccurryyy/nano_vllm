"""Check real model Q/K/V against SDPA reading the SAME quantized cache.

This isolates attention implementation error from quantization discontinuities
caused by slightly different upstream hidden states in two complete models.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from nanovllm import LLM, SamplingParams
from nanovllm.layers.attention import Attention
from nanovllm.utils.context import get_context


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default="int8_per_token_head", choices=["int8_per_token_head", "fp16_k_int8_v"])
    p.add_argument("--output", type=Path, default=Path("results/int8-real-attention.json"))
    args = p.parse_args()
    llm = LLM(str(Path.home() / 'huggingface/Qwen3-0.6B'),
              kv_cache_dtype=args.cache, attention_backend='triton',
              compile_model=False, enforce_eager=True, max_model_len=1024,
              max_num_batched_tokens=128, max_num_seqs=1, gpu_memory_utilization=.7)
    records = []
    for layer, module in enumerate(m for m in llm.model_runner.model.modules() if isinstance(m, Attention)):
        original = module.forward

        def checked(q, k, v, original=original, module=module, layer=layer):
            result = original(q, k, v)
            reference = module.torch_forward(q, k, v, get_context(), already_stored=True)
            torch.testing.assert_close(result, reference, atol=1e-3, rtol=1e-3)
            records.append(dict(layer=layer, prefill=get_context().is_prefill,
                                query_tokens=q.size(0), max_abs_error=(result-reference).abs().max().item()))
            return result

        module.forward = checked
    prompt = llm.tokenizer.encode('你好。' * 200)[:320]
    llm.generate([prompt], SamplingParams(max_tokens=4, ignore_eos=True), use_tqdm=False)
    llm.generate(['介绍一下自己'], SamplingParams(max_tokens=4, ignore_eos=True), use_tqdm=False)
    report = dict(comparisons=len(records), all_within_tolerance=True,
                  max_abs_error=max(r['max_abs_error'] for r in records),
                  tolerance=dict(atol=1e-3, rtol=1e-3), records=records)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'records'}))
    llm.exit()


if __name__ == '__main__':
    main()

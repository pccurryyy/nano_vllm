import os
import argparse
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--prompt", action="append", help="Repeat for multiple prompts")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--backend", choices=["auto", "torch", "triton", "flash"], default="auto")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--kv-cache-dtype", choices=["auto", "int8_per_token_head", "fp16_k_int8_v"], default="auto")
    parser.add_argument("--mixed-batch", action="store_true")
    parser.add_argument("--cpu-kv-blocks", type=int, default=0)
    parser.add_argument("--kv-blocks", type=int, default=-1)
    parser.add_argument("--weight-offload-layers", type=int, default=0)
    parser.add_argument("--weight-window", type=int, default=1)
    parser.add_argument("--stable-fp16", action="store_true", help="Disable reduced-precision FP16 GEMM reductions")
    args = parser.parse_args()
    path = os.path.expanduser(args.model)
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=not args.cuda_graph, tensor_parallel_size=1,
              attention_backend=args.backend, compile_model=args.compile,
              kv_cache_dtype=args.kv_cache_dtype,
              enable_mixed_batch=args.mixed_batch,
              enable_kv_offload=args.cpu_kv_blocks > 0,
              num_cpu_kvcache_blocks=args.cpu_kv_blocks,
              num_kvcache_blocks=args.kv_blocks,
              weight_offload_num_layers=args.weight_offload_layers,
              weight_offload_window_size=args.weight_window,
              fp16_reduced_precision_reduction=False if args.stable_fp16 else None,
              max_model_len=1024, max_num_batched_tokens=256,
              max_num_seqs=2, gpu_memory_utilization=0.7)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=args.max_tokens)
    prompts = args.prompt or [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()

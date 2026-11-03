"""Reproducible workload and backend comparison; no initialization in timing."""
import argparse
import json
import os
import platform
import random
import statistics
from pathlib import Path
from time import perf_counter

import torch
import transformers
import triton
from nanovllm import LLM, SamplingParams
from nanovllm.engine.block_manager import BlockManager


def make_workload(preset, seed):
    rng = random.Random(seed)
    count, minimum, maximum_input, maximum_output = (
        (256, 100, 1024, 1024) if preset == 'readme' else (4, 16, 64, 32)
    )
    prompts = [[rng.randint(0, 10000) for _ in range(rng.randint(minimum, maximum_input))]
               for _ in range(count)]
    lengths = [rng.randint(minimum, maximum_output) for _ in range(count)]
    return prompts, lengths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='~/huggingface/Qwen3-0.6B')
    parser.add_argument('--preset', choices=['local', 'readme'], default='local')
    parser.add_argument('--backend', choices=['auto', 'torch', 'triton', 'flash'], default='auto')
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--max-num-seqs', type=int, default=2)
    parser.add_argument('--max-batched-tokens', type=int, default=256)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.7)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--kv-cache-dtype', choices=['auto', 'int8_per_token_head', 'fp16_k_int8_v'], default='auto')
    parser.add_argument('--requests', type=int)
    parser.add_argument('--input-tokens', type=int)
    parser.add_argument('--output-tokens', type=int)
    parser.add_argument('--kv-blocks', type=int, default=-1, help='Fixed pool size for storage/performance comparison')
    parser.add_argument('--max-model-len', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dry-run', action='store_true', help='Inspect workload without loading the model')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    prompts, lengths = make_workload(args.preset, args.seed)
    overrides = (args.requests, args.input_tokens, args.output_tokens)
    if any(x is not None for x in overrides):
        if any(x is None or x < 1 for x in overrides):
            parser.error('requests, input-tokens and output-tokens must all be positive when overridden')
        rng = random.Random(args.seed)
        prompts = [[rng.randint(0, 10000) for _ in range(args.input_tokens)] for _ in range(args.requests)]
        lengths = [args.output_tokens] * args.requests
    workload = dict(preset=args.preset, seed=args.seed, requests=len(prompts),
                    input_tokens=sum(map(len, prompts)), output_tokens=sum(lengths),
                    max_input_tokens=max(map(len, prompts)), max_output_tokens=max(lengths))
    if args.dry_run:
        print(json.dumps(workload, indent=2))
        return
    torch.manual_seed(args.seed)
    eager = args.eager or args.backend == 'torch'
    init_start = perf_counter()
    llm = LLM(os.path.expanduser(args.model), attention_backend=args.backend,
              kv_cache_dtype=args.kv_cache_dtype,
              num_kvcache_blocks=args.kv_blocks,
              compile_model=args.compile, enforce_eager=eager,
              max_model_len=args.max_model_len,
              max_num_batched_tokens=args.max_batched_tokens,
              max_num_seqs=args.max_num_seqs, gpu_memory_utilization=args.gpu_memory_utilization)
    init_seconds = perf_counter() - init_start
    params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=n) for n in lengths]
    llm.generate(['Benchmark: '], SamplingParams(max_tokens=8, ignore_eos=True), use_tqdm=False)
    # Prime ragged prefill/decode shapes without repeating the entire README
    # workload (133,966 output tokens). Each timed run starts with a cold cache.
    llm.generate(prompts[:args.max_num_seqs], SamplingParams(max_tokens=8, ignore_eos=True), use_tqdm=False)
    timings = []
    actual_tokens = []
    for _ in range(args.repeats):
        llm.scheduler.block_manager = BlockManager(
            llm.model_runner.config.num_kvcache_blocks,
            llm.model_runner.config.kvcache_block_size,
        )
        torch.manual_seed(args.seed)
        torch.cuda.synchronize()
        start = perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        torch.cuda.synchronize()
        seconds = perf_counter() - start
        counts = [len(o['token_ids']) for o in outputs]
        if counts != lengths:
            raise AssertionError(f'Generated token counts differ: {counts} != {lengths}')
        timings.append(seconds)
        actual_tokens.append(sum(counts))
    runner = llm.model_runner
    report = dict(
        workload=workload,
        environment=dict(gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                         torch=torch.__version__, cuda=torch.version.cuda, triton=triton.__version__,
                         transformers=transformers.__version__, python=platform.python_version()),
        config=dict(model=os.path.expanduser(args.model), backend=runner.attention_backend,
                    kv_cache_dtype=runner.config.kv_cache_dtype,
                    cache_data_bytes=runner.cache_bytes, cache_scale_bytes=runner.scale_bytes,
                    compile=runner.compile_model, cuda_graph=not runner.enforce_eager,
                    graph_batches=runner.graph_bs if not runner.enforce_eager else [],
                    max_num_seqs=args.max_num_seqs, max_batched_tokens=args.max_batched_tokens,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    kv_blocks=runner.config.num_kvcache_blocks, dtype=str(runner.config.hf_config.dtype)),
        init_seconds=init_seconds, seconds=timings, actual_output_tokens=actual_tokens,
        tokens_per_second=[n / t for n, t in zip(actual_tokens, timings)],
        median_tokens_per_second=statistics.median(n / t for n, t in zip(actual_tokens, timings)),
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        prefix_cache='cold at the start of each measured repetition',
        comparison_note='Local hardware/configuration; not a reproduction of the RTX 4070 Laptop README throughput.',
    )
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + '\n')
    print(serialized)


if __name__ == '__main__':
    main()

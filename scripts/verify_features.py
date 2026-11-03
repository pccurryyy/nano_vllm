"""End-to-end batching, chunked prefill, prefix-cache and graph checks."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from nanovllm import LLM, SamplingParams
from nanovllm.engine.block_manager import BlockManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=str(Path.home() / 'huggingface/Qwen3-0.6B'))
    parser.add_argument('--backend', choices=['torch', 'triton', 'flash'], default='triton')
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kv-cache-dtype', choices=['auto', 'int8_per_token_head', 'fp16_k_int8_v'], default='auto')
    args = parser.parse_args()
    llm = LLM(args.model, attention_backend=args.backend, compile_model=args.compile,
              kv_cache_dtype=args.kv_cache_dtype,
              enforce_eager=args.eager or args.backend == 'torch',
              max_model_len=1024, max_num_batched_tokens=128,
              max_num_seqs=2, gpu_memory_utilization=0.7)
    events = []
    original_schedule = llm.scheduler.schedule

    def schedule():
        seqs, prefill = original_schedule()
        events.append(dict(prefill=prefill, batch=len(seqs),
                           cached=[s.num_cached_tokens for s in seqs],
                           chunked=any(s.num_cached_tokens + s.num_scheduled_tokens < s.num_prompt_tokens for s in seqs)))
        return seqs, prefill

    llm.scheduler.schedule = schedule
    tokenizer = llm.tokenizer
    prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': p}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
               for p in ['1加1等于多少？', '用中文说一句问候。']]
    params = SamplingParams(temperature=0.00001, max_tokens=24)
    short = llm.generate(prompts, params, use_tqdm=False)
    eos_stop = any(o['token_ids'][-1] == tokenizer.eos_token_id and len(o['token_ids']) < params.max_tokens for o in short)
    assert eos_stop, 'No short request verified EOS termination'
    batched_decode = any(not e['prefill'] and e['batch'] == 2 for e in events)
    # 320 input tokens cross a physical page and exceed the prefill budget.
    ids = tokenizer.encode('你好。' * 200)[:320]
    assert len(ids) == 320
    prefix_start = len(events)
    first = llm.generate([ids], SamplingParams(temperature=0.00001, max_tokens=8, ignore_eos=True), use_tqdm=False)[0]
    chunked = any(e['chunked'] for e in events[prefix_start:])
    reuse_start = len(events)
    second = llm.generate([ids], SamplingParams(temperature=0.00001, max_tokens=8, ignore_eos=True), use_tqdm=False)[0]
    reused = max((max(e['cached']) for e in events[reuse_start:] if e['prefill']), default=0)
    assert first['token_ids'] == second['token_ids'], 'Prefix reuse changed generation'
    assert reused >= 256 and chunked and batched_decode
    assert llm.is_finished() and not llm.scheduler.block_manager.used_block_ids
    # Force real cache pressure while retaining the existing GPU allocation.
    # Two 256-token prompts fill both logical pages; decode must preempt one.
    llm.scheduler.block_manager = BlockManager(2, 256)
    preemptions = []
    original_preempt = llm.scheduler.preempt

    def preempt(seq):
        preemptions.append(seq.seq_id)
        return original_preempt(seq)

    llm.scheduler.preempt = preempt
    pressure_prompts = [ids[:256], tokenizer.encode('早上好。' * 200)[:256]]
    pressure_outputs = llm.generate(pressure_prompts,
        SamplingParams(temperature=0.00001, max_tokens=3, ignore_eos=True), use_tqdm=False)
    assert preemptions and [len(o['token_ids']) for o in pressure_outputs] == [3, 3]
    assert llm.is_finished() and not llm.scheduler.block_manager.used_block_ids
    report = dict(backend=llm.model_runner.attention_backend,
                  kv_cache_dtype=args.kv_cache_dtype,
                  compile=llm.model_runner.compile_model,
                  cuda_graph=not llm.model_runner.enforce_eager,
                  graph_batches=llm.model_runner.graph_bs if not llm.model_runner.enforce_eager else [],
                  batched_decode=batched_decode, chunked_prefill=chunked,
                  eos_termination=eos_stop, output_token_limit=len(first['token_ids']) == 8,
                  reused_prefix_tokens=reused, prefix_generation_equal=True,
                  no_leaked_pages=True, short_outputs=short,
                  preemptions=len(preemptions), preemption_outputs=pressure_outputs,
                  long_output=first, events=events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'events'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

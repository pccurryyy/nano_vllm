"""Same 128 MiB KV budget: compare capacity and preemption under 3 requests."""
import argparse
import json
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from nanovllm import LLM, SamplingParams
from nanovllm.layers.kv_cache import cache_block_bytes, INT8_CACHE, MIXED_CACHE

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--cache', choices=['auto', MIXED_CACHE], required=True)
args = p.parse_args()
budget = 128*2**20
block_bytes = cache_block_bytes(28,256,8,128,torch.float16,False,args.cache == MIXED_CACHE)
llm = LLM(str(Path.home() / 'huggingface/Qwen3-0.6B'), attention_backend='triton',
          compile_model=False, enforce_eager=True, kv_cache_dtype=args.cache,
          max_model_len=512,max_num_batched_tokens=256,max_num_seqs=3,
          gpu_memory_utilization=.7,num_kvcache_blocks=budget//block_bytes)
preemptions = []
original = llm.scheduler.preempt
def preempt(seq):
    preemptions.append(seq.num_tokens)
    return original(seq)
llm.scheduler.preempt = preempt
prompts = [[100+i]*256 for i in range(3)]
out = llm.generate(prompts, SamplingParams(max_tokens=3,ignore_eos=True),use_tqdm=False)
assert [len(o['token_ids']) for o in out] == [3,3,3]
actual = llm.model_runner.cache_bytes+llm.model_runner.scale_bytes
assert actual <= budget and actual == block_bytes*llm.config.num_kvcache_blocks
report = dict(cache=args.cache,budget_bytes=budget,allocated_bytes=actual,
              blocks=llm.config.num_kvcache_blocks,preemptions=len(preemptions),
              output_lengths=[len(o['token_ids']) for o in out],
              free_blocks_after=len(llm.scheduler.block_manager.free_block_ids))
assert report['free_blocks_after'] == report['blocks']
assert len(preemptions) > 0 if args.cache == 'auto' else len(preemptions) == 0
Path(f'results/kv-capacity-{args.cache}.json').write_text(json.dumps(report,indent=2))
print(report)
llm.exit()

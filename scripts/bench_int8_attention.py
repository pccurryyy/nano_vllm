"""CUDA-event timing of one-layer paged decode, including KV writes.

Graph mode isolates repeated GPU execution from Python dispatch. This is a
microbenchmark, not full-model throughput. Run without another GPU workload.
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from nanovllm.layers.attention import Attention
from nanovllm.layers.kv_cache import INT8_CACHE, store_int8
from nanovllm.utils.context import set_context, reset_context


def main():
    torch.manual_seed(456)
    cases = []
    for n in (1, 2):
        for length in (128, 512, 1024, 2048):
            q = torch.randn(n, 16, 128, device='cuda', dtype=torch.float16)
            k = torch.randn(n, 8, 128, device='cuda', dtype=torch.float16)
            v = torch.randn_like(k)
            blocks_per_seq = (length+255)//256
            blocks = n * blocks_per_seq
            source_k = torch.randn(blocks, 256, 8, 128, device='cuda', dtype=torch.float16)
            source_v = torch.randn_like(source_k)
            pages = torch.arange(blocks, device='cuda', dtype=torch.int32).view(n, blocks_per_seq)
            slots = torch.tensor([i*blocks_per_seq*256+length-1 for i in range(n)], device='cuda', dtype=torch.int32)
            lens = torch.full((n,), length, device='cuda', dtype=torch.int32)
            set_context(False, slot_mapping=slots, context_lens=lens, block_tables=pages)
            for fmt in ('auto', INT8_CACHE):
                a = Attention(16, 128, 128 ** -.5, 8)
                a.backend, a.kv_cache_dtype = 'triton', fmt
                if fmt == INT8_CACHE:
                    a.k_cache = torch.empty_like(source_k, dtype=torch.int8)
                    a.v_cache = torch.empty_like(source_v, dtype=torch.int8)
                    a.k_scale_cache = torch.empty(blocks, 256, 8, device='cuda')
                    a.v_scale_cache = torch.empty_like(a.k_scale_cache)
                    store_int8(source_k.flatten(0,1), source_v.flatten(0,1), a.k_cache, a.v_cache,
                               a.k_scale_cache, a.v_scale_cache,
                               torch.arange(blocks*256, device='cuda', dtype=torch.int32))
                else:
                    a.k_cache, a.v_cache = source_k.clone(), source_v.clone()
                a(q, k, v)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = a(q, k, v)
                samples = []
                for _ in range(3):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(20):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end)/20)
                cases.append(dict(requests=n, context=length, cache=fmt, ms=samples,
                                  median_ms=statistics.median(samples)))
                del graph, out, a
    reset_context()
    path = Path('results/int8-attention-benchmark.json')
    path.write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(), cases=cases,
                                   note='One-layer graph replay, CUDA events, includes cache write.'), indent=2))
    print(path)


if __name__ == '__main__':
    main()

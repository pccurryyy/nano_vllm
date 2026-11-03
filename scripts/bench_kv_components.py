"""Paired CUDA-event graphs for KV write/read/combined costs, not model speed."""
import json
import statistics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from nanovllm.layers.attention import Attention, store_kvcache
from nanovllm.layers.kv_cache import INT8_CACHE, MIXED_CACHE, store_int8
from nanovllm.layers.triton_attention import triton_attention
from nanovllm.utils.context import set_context, get_context, reset_context


def main():
    torch.manual_seed(456)
    records = []
    for n in (1, 2):
        for length in (512, 2048):
            q = torch.randn(n, 16, 128, device='cuda', dtype=torch.float16)
            k = torch.randn(n, 8, 128, device='cuda', dtype=torch.float16)
            v = torch.randn_like(k)
            width = (length+255)//256
            blocks = n*width
            source_k = torch.randn(blocks, 256, 8, 128, device='cuda', dtype=torch.float16)
            source_v = torch.randn_like(source_k)
            pages = torch.arange(blocks, device='cuda', dtype=torch.int32).view(n, width)
            slots = torch.tensor([i*width*256+length-1 for i in range(n)], device='cuda', dtype=torch.int32)
            lens = torch.full((n,), length, device='cuda', dtype=torch.int32)
            set_context(False, slot_mapping=slots, context_lens=lens, block_tables=pages)
            graphs, samples = {}, {}
            for fmt in ('auto', INT8_CACHE, MIXED_CACHE):
                a = Attention(16, 128, 128 ** -.5, 8)
                a.backend, a.kv_cache_dtype = 'triton', fmt
                a.k_cache = source_k.clone() if fmt != INT8_CACHE else torch.empty_like(source_k, dtype=torch.int8)
                a.v_cache = source_v.clone() if fmt == 'auto' else torch.empty_like(source_v, dtype=torch.int8)
                if fmt != 'auto':
                    a.v_scale_cache = torch.empty(blocks, 256, 8, device='cuda')
                    a.k_scale_cache = torch.empty_like(a.v_scale_cache) if fmt == INT8_CACHE else None
                    store_int8(source_k.flatten(0,1), source_v.flatten(0,1), a.k_cache, a.v_cache,
                        a.k_scale_cache, a.v_scale_cache, torch.arange(blocks*256, device='cuda', dtype=torch.int32),
                        quantize_key=fmt == INT8_CACHE)
                def write():
                    if fmt == 'auto':
                        store_kvcache(k, v, a.k_cache, a.v_cache, slots)
                    else:
                        store_int8(k, v, a.k_cache, a.v_cache, a.k_scale_cache, a.v_scale_cache,
                                   slots, quantize_key=fmt == INT8_CACHE)
                def read():
                    return triton_attention(q, a.k_cache, a.v_cache, get_context(), a.scale,
                                            256, a.k_scale_cache, a.v_scale_cache)
                for name, call in [('write', write), ('read', read), ('combined', lambda: a(q,k,v))]:
                    for _ in range(5):
                        call()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(64):
                            out = call()
                    # Keep outputs and buffers alive for the captured addresses.
                    graphs[(fmt,name)] = (graph, out, a)
                    samples[(fmt,name)] = []
            keys = list(graphs)
            for repeat in range(7):
                order = keys if repeat % 2 == 0 else list(reversed(keys))
                for key in order:
                    graph = graphs[key][0]
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(10):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    samples[key].append(start.elapsed_time(end)/640)
            for (fmt,name), values in samples.items():
                records.append(dict(requests=n, context=length, cache=fmt, component=name,
                                    ms=values, median_ms=statistics.median(values)))
            del graphs
    reset_context()
    path = Path('results/kv-component-benchmark.json')
    path.write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(), cases=records,
        note='7 alternating-order repeats; 64 operations captured per graph, 10 replays per sample. This avoids host submission gaps for short writers. Synthetic one-layer decode.'),indent=2))
    print(path)


if __name__ == '__main__':
    main()

"""Real-model, fixed-history regression of mixed batching and both offloads.

Run one case per process; never concurrent GPU runs. No speedup claim is made.
Outputs are deliberately forced to token 42 so all comparisons see the same
histories. Attention arithmetic is also checked independently in unit tests.
"""
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from nanovllm import LLM, SamplingParams


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', choices=['auto', 'fp16_k_int8_v'], default='auto')
    p.add_argument('--mixed', action='store_true')
    p.add_argument('--decode-first', action='store_true')
    p.add_argument('--cpu-blocks', type=int, default=0)
    p.add_argument('--weight-layers', type=int, default=0)
    p.add_argument('--weight-window', type=int, default=1)
    p.add_argument('--graph', action='store_true')
    p.add_argument('--blocks', type=int, default=4)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', type=Path)
    p.add_argument('--full-reduction', action='store_true', help='Diagnostic: disable reduced-precision FP16 reductions')
    p.add_argument('--resident-reference', action='store_true',
                   help='Keep 8 logical blocks on GPU but use the offload scheduler working-set limit of 4')
    args = p.parse_args()
    if args.full_reduction:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    llm = LLM(str(Path.home() / 'huggingface/Qwen3-0.6B'), attention_backend='triton',
              compile_model=False, enforce_eager=not args.graph,
              kv_cache_dtype=args.cache, max_model_len=768, max_num_batched_tokens=256,
              max_num_seqs=3, num_kvcache_blocks=args.blocks, gpu_memory_utilization=.7,
              enable_mixed_batch=args.mixed, enable_kv_offload=args.cpu_blocks > 0,
              num_cpu_kvcache_blocks=args.cpu_blocks, weight_offload_num_layers=args.weight_layers,
              weight_offload_window_size=args.weight_window,
              fp16_reduced_precision_reduction=False if args.full_reduction else None,
              scheduling_policy='decode_first' if args.decode_first else 'prefill_first')
    if args.resident_reference:
        if args.blocks != 8 or args.cpu_blocks:
            raise ValueError('Resident reference uses 8 GPU blocks and no CPU blocks')
        # Test-only control: same logical capacity and scheduling bound as a
        # 4-GPU/4-CPU pool, but all bytes remain resident. Isolates transport
        # from preemption/recomputation and batch-shape numerical differences.
        llm.scheduler.offload = True
        llm.scheduler.block_manager.num_gpu_blocks = 4
    labels, active, logits = {}, [], {i: [] for i in range(3)}
    original = llm.model_runner.run

    def run(seqs, phase):
        active[:] = seqs
        return original(seqs, phase)

    def sample(values, temperatures):
        for i, seq in enumerate(active):
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                logits[labels[seq.seq_id]].append(values[i].float().cpu())
        return torch.full((values.size(0),), 42, dtype=torch.long, device=values.device)

    llm.model_runner.run = run
    llm.model_runner.sampler.forward = sample
    for i in range(3):
        seq = llm.add_request([100 + i] * 256, SamplingParams(max_tokens=6, ignore_eos=True))
        labels[seq.seq_id] = i
    steps, mixed_steps, graph_steps = 0, 0, 0
    torch.cuda.synchronize()
    start = perf_counter()
    while not llm.is_finished():
        llm.step()
        mixed_steps += llm.last_step_metrics['mixed']
        graph_steps += (not llm.model_runner.enforce_eager and llm.last_step_metrics['prefill_tokens'] == 0)
        steps += 1
        assert steps < 100, 'scheduler failed to finish'
    torch.cuda.synchronize()
    wall = perf_counter() - start
    data = torch.stack([torch.stack(logits[i]) for i in range(3)])
    assert data.shape[:2] == (3, 6)
    assert len(llm.scheduler.block_manager.free_block_ids) == args.blocks + args.cpu_blocks
    report = dict(config=vars(args).copy(), steps=steps, mixed_steps=mixed_steps, graph_decode_steps=graph_steps,
                  forced_output_lengths=[len(logits[i]) for i in range(3)], wall_seconds=wall,
                  preemptions=llm.scheduler.preemptions, all_logical_blocks_returned=True,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  effective_eager=llm.model_runner.enforce_eager)
    report['fp16_reduced_precision_reduction'] = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    report['environment'] = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                                 cuda=torch.version.cuda, backend=llm.model_runner.attention_backend)
    report['request_metrics'] = [llm.get_request_metrics(seq_id) for seq_id in labels]
    report['config'] = {k: str(v) if isinstance(v, Path) else v for k, v in report['config'].items()}
    if hasattr(llm.model_runner, 'kv_transfer'):
        transfer = llm.model_runner.kv_transfer
        report.update(kv_transfer=transfer.stats, cpu_pool_bytes=transfer.cpu_bytes,
                      cpu_scratch_bytes=transfer.scratch_bytes)
        assert transfer.stats['h2d'] > 0 and transfer.stats['d2h'] > 0
    if hasattr(llm.model_runner, 'weight_offload'):
        manager = llm.model_runner.weight_offload
        report.update(weight_transfer=manager.stats, cpu_weight_bytes=manager.cpu_bytes)
        assert manager.stats['peak_active_layers'] <= args.weight_window
        assert not manager.active
    if args.reference:
        reference = torch.load(args.reference.with_suffix('.pt'), weights_only=True)
        delta = (data - reference).abs()
        report['logits_max_abs'] = delta.max().item()
        report['argmax_agreement'] = (data.argmax(-1) == reference.argmax(-1)).float().mean().item()
        try:
            # Existing full-model acceptance; do not relax it after seeing data.
            torch.testing.assert_close(data, reference, atol=.1, rtol=.01)
            report['strict_logits_pass'] = True
        except AssertionError:
            report['strict_logits_pass'] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.output.with_suffix('.pt'))
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    llm.exit()
    if report.get('strict_logits_pass') is False:
        raise AssertionError('Full-model logits exceeded the unchanged strict tolerance; see report')


if __name__ == '__main__':
    main()

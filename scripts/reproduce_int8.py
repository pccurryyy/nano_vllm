"""Sequential, isolated GPU validation and fixed-slot FP16/INT8 comparisons."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'


def run(name, script, args):
    print(name, flush=True)
    result = RESULTS / f'{name}.json'
    with (RESULTS / f'{name}.log').open('w') as log:
        subprocess.run([sys.executable, script, *args, '--output', str(result)],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads(result.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--skip-quality', action='store_true', help='Use already completed quality reports')
    p.add_argument('--resume', action='store_true', help='Reuse completed functional/numerical reports; rerun benchmarks')
    args = p.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if not args.resume:
        with (RESULTS / 'int8-unit-tests.txt').open('w') as log:
            subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    else:
        assert (RESULTS / 'int8-unit-tests.txt').read_text().rstrip().endswith('OK')
    features = []
    for cache in ('auto', 'int8_per_token_head'):
        for mode in ('eager', 'graph'):
            options = ['--backend', 'triton', '--kv-cache-dtype', cache]
            options += ['--eager'] if mode == 'eager' else ['--compile']
            name = f'int8-regression-{cache}-{mode}'
            features.append(json.loads((RESULTS / f'{name}.json').read_text()) if args.resume
                            else run(name, 'scripts/verify_features.py', options))
    for result in features:
        assert all(result[k] for k in ['batched_decode', 'chunked_prefill', 'eos_termination',
                                      'output_token_limit', 'prefix_generation_equal', 'no_leaked_pages'])
    if not args.skip_quality:
        for cache in ('auto', 'int8_per_token_head'):
            with (RESULTS / 'int8-eval' / f'{cache}.log').open('w') as log:
                subprocess.run([sys.executable, 'scripts/validate_int8_quality.py', '--cache', cache],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    reference = RESULTS / 'int8-reference.pt'
    if not args.resume:
        run('int8-reference-logits', 'scripts/verify_model_logits.py',
            ['--backend', 'torch', '--kv-cache-dtype', 'int8_per_token_head',
             '--reference', str(reference), '--save-reference'])
        run('int8-triton-logits', 'scripts/verify_model_logits.py',
            ['--backend', 'triton', '--kv-cache-dtype', 'int8_per_token_head',
             '--reference', str(reference), '--measure-only'])
        subprocess.run([sys.executable, 'scripts/check_int8_model_attention.py'], cwd=ROOT, check=True)
    measurements = []
    for case in ('local', '512', '2048'):
        for cache in ('auto', 'int8_per_token_head'):
            options = ['--backend', 'triton', '--compile', '--kv-cache-dtype', cache,
                       '--kv-blocks', '28', '--repeats', '3', '--gpu-memory-utilization', '0.85']
            if case != 'local':
                options += ['--requests', '1', '--input-tokens', case, '--output-tokens', '32']
            measurements.append(run(f'int8-bench-{case}-{cache}', 'bench.py', options))
    subprocess.run([sys.executable, 'scripts/bench_int8_attention.py'], cwd=ROOT, check=True)
    quality = json.loads((RESULTS / 'int8-eval' / 'int8_per_token_head.json').read_text())
    comparisons = []
    for i in range(0, len(measurements), 2):
        base, quant = measurements[i:i+2]
        assert base['workload'] == quant['workload']
        assert base['actual_output_tokens'] == quant['actual_output_tokens']
        assert base['config']['kv_blocks'] == quant['config']['kv_blocks'] == 28
        bbytes = base['config']['cache_data_bytes']
        qbytes = quant['config']['cache_data_bytes'] + quant['config']['cache_scale_bytes']
        assert qbytes / bbytes == .515625
        comparisons.append(dict(workload=base['workload'], fp16_tps=base['median_tokens_per_second'],
                                int8_tps=quant['median_tokens_per_second'], storage_ratio=qbytes/bbytes,
                                throughput_ratio=quant['median_tokens_per_second']/base['median_tokens_per_second']))
    numerical = json.loads((RESULTS / 'int8-triton-logits.json').read_text())
    matched_attention = json.loads((RESULTS / 'int8-real-attention.json').read_text())
    summary = dict(reference_commit='23fbc2ec2f7219888bfc76cf987381f501151971',
                   functional_checks_passed=True, quality=quality, model_reference_comparison=numerical,
                   matched_cache_attention=dict(comparisons=matched_attention['comparisons'],
                                                all_within_tolerance=matched_attention['all_within_tolerance'],
                                                max_abs_error=matched_attention['max_abs_error']),
                   benchmarks=comparisons,
                   release_status='quality-gated' if (quality['quality_gates_passed']
                                                      and numerical['logits_within_tolerance']
                                                      and matched_attention['all_within_tolerance']) else 'experimental')
    (RESULTS / 'int8-summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(dict(quality_gates_passed=quality['quality_gates_passed'],
                         comparisons=comparisons), indent=2), flush=True)


if __name__ == '__main__':
    main()

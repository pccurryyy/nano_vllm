"""Sequential GPU validation: mixed KV, frozen heldout quality and capacity."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'


def run(name, *args):
    print(name, flush=True)
    suffix = '.txt' if name == 'mixed-unit-tests' else '.log'
    with (RESULTS / f'{name}{suffix}').open('w') as log:
        subprocess.run([sys.executable,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reuse-baseline',action='store_true',help='Use an existing matching frozen FP16 heldout baseline')
    p.add_argument('--summarize-only',action='store_true',help='Validate and summarize completed evidence without rerunning GPU jobs')
    args = p.parse_args()
    if args.summarize_only:
        summarize()
        return
    RESULTS.mkdir(exist_ok=True)
    run('mixed-unit-tests','-m','unittest','discover','-s','tests','-v')
    for fmt in ('auto','fp16_k_int8_v'):
        if fmt == 'auto' and args.reuse_baseline:
            assert (RESULTS/'mixed-heldout/auto.json').exists()
            continue
        run(f'mixed-heldout-{fmt}','scripts/validate_int8_quality.py','--cache',fmt,
            '--heldout','--directory','results/mixed-heldout')
    for fmt in ('auto','fp16_k_int8_v'):
        run(f'mixed-heldout-compiled-{fmt}','scripts/validate_int8_quality.py','--cache',fmt,
            '--heldout','--compile','--directory','results/mixed-heldout-compiled')
    features = []
    for fmt in ('auto','fp16_k_int8_v'):
        for mode in ('eager','graph'):
            name = f'mixed-regression-{fmt}-{mode}'
            run(name,'scripts/verify_features.py','--kv-cache-dtype',fmt,
                '--eager' if mode == 'eager' else '--compile','--output',f'results/{name}.json')
            result = json.loads((RESULTS/f'{name}.json').read_text())
            assert all(result[k] for k in ['batched_decode','chunked_prefill','eos_termination',
                                          'output_token_limit','prefix_generation_equal','no_leaked_pages'])
            features.append(result)
    reference = 'results/mixed-reference.pt'
    # Retain the original compiled-vs-uncompiled failure as diagnostic evidence.
    run('mixed-torch-logits','scripts/verify_model_logits.py','--backend','torch',
        '--kv-cache-dtype','fp16_k_int8_v','--reference',reference,'--save-reference',
        '--output','results/mixed-torch-logits.json')
    run('mixed-triton-logits','scripts/verify_model_logits.py','--backend','triton','--compile',
        '--kv-cache-dtype','fp16_k_int8_v','--reference',reference,'--measure-only',
        '--output','results/mixed-triton-logits.json')
    run('mixed-triton-uncompiled-logits','scripts/verify_model_logits.py','--backend','triton',
        '--kv-cache-dtype','fp16_k_int8_v','--reference',reference,'--measure-only',
        '--output','results/mixed-triton-uncompiled-logits.json')
    matched_reference = 'results/mixed-compiled-reference.pt'
    run('mixed-torch-compiled-logits','scripts/verify_model_logits.py','--backend','torch','--compile',
        '--kv-cache-dtype','fp16_k_int8_v','--reference',matched_reference,'--save-reference',
        '--output','results/mixed-torch-compiled-logits.json')
    run('mixed-triton-matched-compile-logits','scripts/verify_model_logits.py','--backend','triton','--compile',
        '--kv-cache-dtype','fp16_k_int8_v','--reference',matched_reference,'--measure-only',
        '--output','results/mixed-triton-matched-compile-logits.json')
    run('mixed-real-attention','scripts/check_int8_model_attention.py','--cache','fp16_k_int8_v',
        '--output','results/mixed-real-attention.json')
    run('mixed-cache-audit','scripts/audit_kv_model_difference.py')
    run('kv-components','scripts/bench_kv_components.py')
    for fmt in ('auto','fp16_k_int8_v'):
        run(f'kv-capacity-{fmt}','scripts/verify_kv_capacity.py','--cache',fmt)
    measurements = []
    for case in ('local','512','2048'):
        for fmt in ('auto','fp16_k_int8_v'):
            name = f'mixed-bench-{case}-{fmt}'
            run(name,'bench.py','--backend','triton','--compile','--kv-cache-dtype',fmt,
                '--kv-blocks','28','--repeats','3','--gpu-memory-utilization','0.85',
                '--output',f'results/{name}.json',
                *([] if case == 'local' else ['--requests','1','--input-tokens',case,'--output-tokens','32']))
            measurements.append(json.loads((RESULTS/f'{name}.json').read_text()))
    summarize()


def summarize():
    measurements = [json.loads((RESULTS/f'mixed-bench-{case}-{fmt}.json').read_text())
                    for case in ('local','512','2048') for fmt in ('auto','fp16_k_int8_v')]
    comparisons = []
    for b,q in zip(measurements[::2], measurements[1::2]):
        assert b['config']['kv_blocks'] == q['config']['kv_blocks'] == 28
        assert b['workload'] == q['workload'] and b['actual_output_tokens'] == q['actual_output_tokens']
        ratio = (q['config']['cache_data_bytes']+q['config']['cache_scale_bytes'])/b['config']['cache_data_bytes']
        assert ratio == .7578125
        comparisons.append(dict(workload=b['workload'],fp16_tps=b['median_tokens_per_second'],
            mixed_tps=q['median_tokens_per_second'],throughput_ratio=q['median_tokens_per_second']/b['median_tokens_per_second'],
            storage_ratio=ratio))
    quality = json.loads((RESULTS/'mixed-heldout/fp16_k_int8_v.json').read_text())
    compiled_quality = json.loads((RESULTS/'mixed-heldout-compiled/fp16_k_int8_v.json').read_text())
    numerical = json.loads((RESULTS/'mixed-triton-matched-compile-logits.json').read_text())
    original_pair = json.loads((RESULTS/'mixed-triton-logits.json').read_text())
    attention = json.loads((RESULTS/'mixed-real-attention.json').read_text())
    assert (RESULTS/'mixed-unit-tests.txt').read_text().rstrip().endswith('OK')
    for fmt in ('auto','fp16_k_int8_v'):
        for mode in ('eager','graph'):
            result = json.loads((RESULTS/f'mixed-regression-{fmt}-{mode}.json').read_text())
            assert all(result[k] for k in ['batched_decode','chunked_prefill','eos_termination',
                                          'output_token_limit','prefix_generation_equal','no_leaked_pages'])
    assert quality['manifest'] == compiled_quality['manifest']
    summary = dict(quality=quality,compiled_quality=compiled_quality,
                   real_input_quantization_audit=json.loads((RESULTS/'mixed-quantization-audit.json').read_text()),model_reference_comparison=numerical,
                   original_cross_compile_comparison=original_pair,
                   capacity=[json.loads((RESULTS/f'kv-capacity-{fmt}.json').read_text()) for fmt in ('auto','fp16_k_int8_v')],
                   same_cache_attention=attention,benchmarks=comparisons,
                   functional_checks_passed=True,
                   matched_configuration_gates_passed=quality['quality_gates_passed'] and compiled_quality['quality_gates_passed'] and numerical['logits_within_tolerance'],
                   original_all_gates_passed=quality['quality_gates_passed'] and compiled_quality['quality_gates_passed'] and original_pair['logits_within_tolerance'],
                   release_status='experimental')
    (RESULTS/'mixed-summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(dict(matched_configuration_gates_passed=summary['matched_configuration_gates_passed'],benchmarks=comparisons),indent=2),flush=True)


if __name__ == '__main__':
    main()

"""Run GPU workloads sequentially and save evidence under results/."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'


def run(name, script, options):
    output = RESULTS / (name + '.json')
    command = [sys.executable, script, *options, '--output', str(output)]
    print('Running:', ' '.join(command), flush=True)
    with (RESULTS / (name + '.log')).open('w') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads(output.read_text())


def main():
    RESULTS.mkdir(exist_ok=True)
    feature_results = []
    for name, options in [
        ('features-torch', ['--backend', 'torch', '--eager', '--no-compile']),
        ('features-triton', ['--backend', 'triton', '--eager', '--no-compile']),
        ('features-graph', ['--backend', 'triton', '--compile']),
    ]:
        feature_results.append(run(name, 'scripts/verify_features.py', options))
    reference = feature_results[0]
    reference_short = [o['token_ids'] for o in reference['short_outputs']]
    for result in feature_results[1:]:
        assert result['long_output']['token_ids'] == reference['long_output']['token_ids'], 'Cached/chunked generation differs from reference'
    # Small FP16 logit differences can change near-tied argmax tokens. Compare
    # logits with identical forced histories instead of requiring all text to
    # match bit for bit across different attention implementations.
    with tempfile.TemporaryDirectory(prefix='nanovllm-logits-') as directory:
        ref_path = str(Path(directory) / 'reference.pt')
        common = ['--reference', ref_path]
        run('logits-torch', 'scripts/verify_model_logits.py',
            ['--backend', 'torch', '--eager', '--no-compile', '--save-reference', *common])
        logits = []
        for name, options in [('logits-triton', ['--eager', '--no-compile']),
                              ('logits-graph', ['--compile'])]:
            logits.append(run(name, 'scripts/verify_model_logits.py', ['--backend', 'triton', *options, *common]))
    bench_results = []
    for name, options in [
        ('torch-eager', ['--backend', 'torch', '--eager', '--no-compile']),
        ('triton-eager', ['--backend', 'triton', '--eager', '--no-compile']),
        ('triton-compiled', ['--backend', 'triton', '--eager', '--compile']),
        ('triton-graph', ['--backend', 'triton', '--compile']),
    ]:
        result = run(name, 'bench.py', [*options, '--repeats', '3'])
        bench_results.append(dict(name=name, tokens_per_second=result['median_tokens_per_second']))
    summary = dict(logits_within_tolerance=all(r['logits_within_tolerance'] for r in logits),
                   preemption_verified=all(r['preemptions'] > 0 for r in feature_results),
                   long_generation_matches_reference=True,
                   short_generation_matches_reference=all([o['token_ids'] for o in r['short_outputs']] == reference_short for r in feature_results[1:]),
                   benchmarks=bench_results,
                   note='Small local workload; not the README RTX 4070 benchmark.')
    (RESULTS / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

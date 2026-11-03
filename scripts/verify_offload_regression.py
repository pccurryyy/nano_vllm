"""Final sequential unit, original-feature, and weight-window regressions."""
import json
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'results/offload-integration-final'
    output.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    cases = [
        ('unit-tests', [python, '-m', 'unittest', 'discover', '-s', 'tests', '-v']),
        ('original-features-fp16', [python, 'scripts/verify_features.py', '--backend', 'triton',
                                  '--output', str(output / 'original-features-fp16.json')]),
        ('original-features-mixed', [python, 'scripts/verify_features.py', '--backend', 'triton',
                                  '--kv-cache-dtype', 'fp16_k_int8_v',
                                  '--output', str(output / 'original-features-mixed.json')]),
    ]
    for name, layers in [('weight-window', 2), ('all-weight-window', 28)]:
        cases.append((name, [python, 'scripts/verify_offload_integration.py',
            '--full-reduction', '--weight-layers', str(layers),
            '--reference', 'results/offload-integration-stable/fp16-baseline.json',
            '--output', str(output / f'{name}.json')]))
    outcomes = []
    for name, command in cases:
        print(f'Running {name}', flush=True)
        with (output / f'{name}.log').open('w') as log:
            proc = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        outcomes.append(dict(case=name, exit_code=proc.returncode))
        print(outcomes[-1], flush=True)
    (output / 'validation.json').write_text(json.dumps(outcomes, indent=2))
    if any(r['exit_code'] for r in outcomes):
        raise SystemExit('Regression failure; inspect preserved logs')


if __name__ == '__main__':
    main()

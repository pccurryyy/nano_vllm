"""Sequential GPU correctness matrix. Includes baselines; no timing claims."""
import json
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-reduction', action='store_true')
    parser.add_argument('--directory', type=Path, default=Path('results/offload-integration'))
    parser.add_argument('--matched-schedule', action='store_true',
                        help='Compare paging against an all-resident control with identical scheduling')
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    results = root / options.directory
    results.mkdir(parents=True, exist_ok=True)
    cases = [
        ('fp16-baseline', []),
        ('fp16-mixed', ['--mixed']),
        ('fp16-decode-first', ['--decode-first']),
        ('fp16-offload', ['--cpu-blocks', '4']),
        ('fp16-mixed-offload-graph', ['--mixed', '--cpu-blocks', '4', '--graph']),
        ('fp16-weight-window', ['--weight-layers', '2', '--weight-window', '1']),
        ('fp16-all-weights-window', ['--weight-layers', '28', '--weight-window', '1']),
        ('mixed-baseline', ['--cache', 'fp16_k_int8_v']),
        ('mixed-mixed', ['--cache', 'fp16_k_int8_v', '--mixed']),
        ('mixed-offload', ['--cache', 'fp16_k_int8_v', '--cpu-blocks', '4']),
        ('mixed-mixed-offload-graph', ['--cache', 'fp16_k_int8_v', '--mixed', '--cpu-blocks', '4', '--graph']),
        ('mixed-all-features', ['--cache', 'fp16_k_int8_v', '--mixed', '--cpu-blocks', '4', '--weight-layers', '2']),
    ]
    matched_references = {}
    if options.matched_schedule:
        resident = ['--blocks', '8', '--resident-reference']
        mixed = ['--cache', 'fp16_k_int8_v']
        cases = [
            ('fp16-resident-reference', resident),
            ('fp16-offload', ['--cpu-blocks', '4']),
            ('fp16-mixed-resident-reference', [*resident, '--mixed']),
            ('fp16-mixed-offload-graph', ['--mixed', '--cpu-blocks', '4', '--graph']),
            ('mixed-resident-reference', [*mixed, *resident]),
            ('mixed-offload', [*mixed, '--cpu-blocks', '4']),
            ('mixed-mixed-resident-reference', [*mixed, *resident, '--mixed']),
            ('mixed-mixed-offload-graph', [*mixed, '--mixed', '--cpu-blocks', '4', '--graph']),
            ('mixed-all-features', [*mixed, '--mixed', '--cpu-blocks', '4', '--weight-layers', '2']),
        ]
        matched_references = {
            'fp16-offload': 'fp16-resident-reference',
            'fp16-mixed-offload-graph': 'fp16-mixed-resident-reference',
            'mixed-offload': 'mixed-resident-reference',
            'mixed-mixed-offload-graph': 'mixed-mixed-resident-reference',
            'mixed-all-features': 'mixed-mixed-resident-reference',
        }
    statuses = []
    for name, args in cases:
        command = [sys.executable, str(root / 'scripts/verify_offload_integration.py'),
                   '--output', str(results / f'{name}.json'), *args]
        if options.full_reduction:
            command += ['--full-reduction']
        reference = None
        if options.matched_schedule:
            reference = matched_references.get(name)
        elif not name.endswith('baseline'):
            reference = 'mixed-baseline' if '--cache' in args else 'fp16-baseline'
        if reference:
            command += ['--reference', str(results / f'{reference}.json')]
        print(f'Running {name}', flush=True)
        with (results / f'{name}.log').open('w') as log:
            proc = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        statuses.append(dict(case=name, exit_code=proc.returncode))
        print(f'{name}: exit={proc.returncode}', flush=True)
    (results / 'matrix.json').write_text(json.dumps(statuses, indent=2))
    if any(x['exit_code'] for x in statuses):
        raise SystemExit('Some cases failed; inspect logs and preserve failure evidence')


if __name__ == '__main__':
    main()

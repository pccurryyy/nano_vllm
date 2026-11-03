"""Audit real writes and compare quantized bytes from independent model runs."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--from-existing', action='store_true', help='Summarize already completed audit tensors')
    args = p.parse_args()
    if not args.from_existing:
        for backend in ('torch', 'triton'):
            name = f'mixed-audit-{backend}'
            with (ROOT / f'results/{name}.log').open('w') as log:
                subprocess.run([sys.executable,'scripts/verify_model_logits.py','--backend',backend,'--eager',
                    '--kv-cache-dtype','fp16_k_int8_v','--audit-cache',f'results/{name}.pt',
                    '--output',f'results/{name}.json','--reference','results/mixed-audit-reference.pt',
                    *(['--save-reference'] if backend == 'torch' else ['--measure-only'])],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    base = torch.load(ROOT / 'results/mixed-audit-torch.pt',weights_only=True)
    actual = torch.load(ROOT / 'results/mixed-audit-triton.pt',weights_only=True)
    assert len(base) == len(actual)
    for backend in ('torch','triton'):
        checks = json.loads((ROOT / f'results/mixed-audit-{backend}.json').read_text())
        assert checks['cache_bytes_match_independent_spec']
    rows = []
    for a,b in zip(base,actual):
        assert a['layer'] == b['layer'] and a['value'].shape == b['value'].shape
        diff = a['value'].float()-b['value'].float()
        rows.append(dict(layer=a['layer'],prefill=a['prefill'],elements=diff.numel(),
            differing_input_elements=int((diff != 0).sum()),
            different_quantized_bytes=int((a['stored_v'] != b['stored_v']).sum()),
            max_input_abs_error=float(diff.abs().max())))
    report = dict(calls=len(rows),cache_bytes_match_independent_spec=True,
        total_elements=sum(r['elements'] for r in rows),
        differing_input_elements=sum(r['differing_input_elements'] for r in rows),
        different_quantized_bytes=sum(r['different_quantized_bytes'] for r in rows),records=rows,
        note='Each backend matches exact V quantization and FP16 K on its own real inputs. Different upstream inputs may cross quantization thresholds.')
    (ROOT / 'results/mixed-quantization-audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'records'},indent=2))


if __name__ == '__main__':
    main()

"""Compare full-model logits with identical teacher-forced decode history."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from nanovllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['torch', 'triton'], required=True)
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--save-reference', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kv-cache-dtype', choices=['auto', 'int8_per_token_head', 'fp16_k_int8_v'], default='auto')
    parser.add_argument('--measure-only', action='store_true', help='Report failed tolerance checks without declaring a pass')
    parser.add_argument('--audit-cache', type=Path, help='Save independent real-input write checks; requires eager execution')
    args = parser.parse_args()
    llm = LLM(str(Path.home() / 'huggingface/Qwen3-0.6B'), attention_backend=args.backend,
              kv_cache_dtype=args.kv_cache_dtype,
              compile_model=args.compile, enforce_eager=args.eager or args.backend == 'torch',
              max_model_len=1024, max_num_batched_tokens=128, max_num_seqs=2,
              gpu_memory_utilization=0.7)
    audits = []
    if args.audit_cache:
        if not llm.model_runner.enforce_eager or args.kv_cache_dtype != 'fp16_k_int8_v':
            raise ValueError('Cache audit requires mixed cache and eager execution')
        from nanovllm.layers.attention import Attention
        from nanovllm.layers.kv_cache import quantize_reference
        from nanovllm.utils.context import get_context
        for layer, module in enumerate(m for m in llm.model_runner.model.modules() if isinstance(m, Attention)):
            def audit(module, inputs, output, layer=layer):
                _, key, value = inputs
                slots = get_context().slot_mapping.long()
                valid = slots >= 0
                actual_k = module.k_cache.flatten(0,1)[slots[valid]]
                actual_v = module.v_cache.flatten(0,1)[slots[valid]]
                actual_s = module.v_scale_cache.flatten(0,1)[slots[valid]]
                expected_v, expected_s = quantize_reference(value[valid])
                torch.testing.assert_close(actual_k, key[valid], atol=0, rtol=0)
                torch.testing.assert_close(actual_v, expected_v, atol=0, rtol=0)
                torch.testing.assert_close(actual_s, expected_s, atol=1e-8, rtol=1e-6)
                audits.append(dict(layer=layer, prefill=get_context().is_prefill,
                    value=value[valid].detach().cpu(), stored_v=actual_v.cpu(), scale=actual_s.cpu()))
            module.register_forward_hook(audit)
    tokenizer = llm.tokenizer
    prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': p}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
               for p in ['1加1等于多少？', '用中文说一句问候。']]
    targets = [tokenizer.encode(t, add_special_tokens=False) + [tokenizer.eos_token_id]
               for t in ['1加1等于2。这是一个基本的数学运算。', '您好！祝您一切顺利！']]
    labels = {}
    trace = []
    current = []
    original_run = llm.model_runner.run

    def run(seqs, is_prefill):
        current[:] = seqs
        return original_run(seqs, is_prefill)

    def sample(logits, temperatures):
        keys = [(labels[s.seq_id][0], s.num_cached_tokens, s.num_completion_tokens) for s in current]
        trace.append((keys, logits.detach().float().cpu()))
        return torch.tensor([labels[s.seq_id][1][s.num_completion_tokens] for s in current], device=logits.device)

    llm.model_runner.run = run
    llm.model_runner.sampler.forward = sample
    for i, (prompt, target) in enumerate(zip(prompts, targets)):
        llm.add_request(prompt, SamplingParams(max_tokens=len(target)))
        labels[llm.scheduler.waiting[-1].seq_id] = (i, target)
    while not llm.is_finished():
        llm.step()
    ids = tokenizer.encode('你好。' * 200)[:320]
    target = tokenizer.encode('你好。你好。', add_special_tokens=False) + [tokenizer.eos_token_id]
    llm.add_request(ids, SamplingParams(max_tokens=len(target)))
    labels[llm.scheduler.waiting[-1].seq_id] = (2, target)
    while not llm.is_finished():
        llm.step()
    report = dict(backend=args.backend, compile=args.compile, kv_cache_dtype=args.kv_cache_dtype,
                  cuda_graph=not llm.model_runner.enforce_eager, steps=len(trace))
    if args.save_reference:
        torch.save(trace, args.reference)
        report['reference_saved'] = True
    else:
        reference = torch.load(args.reference, weights_only=True)
        assert len(trace) == len(reference)
        maximum, error_sum, squared_sum, elements = 0., 0., 0., 0
        agreement, rows, near_ties = 0, 0, 0
        within_tolerance = True
        for (keys, logits), (ref_keys, ref) in zip(trace, reference):
            assert keys == ref_keys, 'Teacher-forced scheduling differs'
            difference = (logits - ref).abs()
            maximum = max(maximum, difference.max().item())
            error_sum += difference.sum().item()
            squared_sum += difference.square().sum().item()
            elements += difference.numel()
            try:
                torch.testing.assert_close(logits, ref, atol=0.1, rtol=0.01)
            except AssertionError:
                within_tolerance = False
                if not args.measure_only:
                    raise
            for row, ref_row, delta in zip(logits, ref, difference):
                rows += 1
                a, b = row.argmax().item(), ref_row.argmax().item()
                if a == b:
                    agreement += 1
                else:
                    margin = (ref_row[b] - ref_row[a]).item()
                    if not args.measure_only:
                        assert margin <= 2 * delta.max().item(), 'Argmax change exceeds numerical error bound'
                    near_ties += 1
        report.update(max_abs_logit_error=maximum, mean_abs_logit_error=error_sum/elements,
                      rms_logit_error=(squared_sum/elements)**0.5,
                      top1_agreement=agreement/rows, near_tie_top1_changes=near_ties,
                      tolerance=dict(atol=0.1, rtol=0.01), logits_within_tolerance=within_tolerance)
    if args.audit_cache:
        torch.save(audits, args.audit_cache)
        report['real_input_cache_write_checks'] = len(audits)
        report['cache_bytes_match_independent_spec'] = True
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

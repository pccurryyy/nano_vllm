"""Frozen evaluation: teacher-forced NLL and 100 forced-choice retrieval cases.

Run --cache auto first, then --cache int8_per_token_head against its tensors.
Quality gates are fixed before evaluation, not inherited from FP16 kernel tests.
The retrieval task scores digit candidates, not unrestricted generated answers.
"""
import argparse
import hashlib
import json
import math
import random
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from nanovllm import LLM, SamplingParams
from nanovllm.engine.block_manager import BlockManager


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', choices=['auto', 'int8_per_token_head', 'fp16_k_int8_v'], required=True)
    p.add_argument('--directory', type=Path, default=Path('results/int8-eval'))
    p.add_argument('--compile', action='store_true', help='Validate compiled layers with CUDA Graph enabled')
    p.add_argument('--heldout', action='store_true', help='New data and seed; never used by development ablations')
    args = p.parse_args()
    root = args.directory
    root.mkdir(parents=True, exist_ok=True)
    if not (root / 'alice.txt').exists():
        raw = urllib.request.urlopen('https://www.gutenberg.org/files/11/11-0.txt', timeout=30).read().decode('utf-8-sig')
        text = raw.split('*** START OF THE PROJECT GUTENBERG EBOOK', 1)[1].split('\n', 1)[1]
        (root / 'alice.txt').write_text(text.split('*** END OF THE PROJECT GUTENBERG EBOOK', 1)[0])
    llm = LLM(str(Path.home() / 'huggingface/Qwen3-0.6B'),
              kv_cache_dtype=args.cache, attention_backend='triton',
              compile_model=args.compile, enforce_eager=not args.compile, max_model_len=2048,
              max_num_batched_tokens=256, max_num_seqs=1, gpu_memory_utilization=.7)
    tok = llm.tokenizer
    texts = {
        'english_public_domain': (root / 'alice.txt').read_text(),
        'chinese_synthetic': ('推理引擎接收请求后，先处理输入，再逐个生成输出。'
                             '每个请求保存自己的序列位置，缓存记录已经计算的键和值。'
                             '共享前缀可以减少重复计算，但后续私有内容必须独立保存。'
                             '性能实验需要固定输入、输出数量和执行条件，记录全部测量结果。\n') * 80,
        'code_repository': '\n'.join((Path('nanovllm') / f).read_text() for f in
                         ['engine/scheduler.py', 'engine/sequence.py', 'models/qwen3.py']) * 3,
    }
    limits = {'english_public_domain': 8192, 'chinese_synthetic': 2048, 'code_repository': 2048}
    if args.heldout:
        texts['chinese_synthetic'] = ('清晨的城市逐渐苏醒，公交车沿着河岸驶过。'
            '图书馆管理员整理归还的书籍，学生在窗边讨论数学问题。'
            '气象站记录温度和降雨，农民根据季节安排播种。'
            '旅行者记录途中见闻，并把照片寄给远方的朋友。\n') * 100
        texts['code_repository'] = '\n'.join((Path('nanovllm') / f).read_text() for f in
            ['layers/linear.py', 'layers/embed_head.py', 'utils/loader.py']) * 6
    corpus = {name: tok.encode(text, add_special_tokens=False)[:limits[name]+1]
              for name, text in texts.items()}
    if args.heldout:
        english = tok.encode(texts['english_public_domain'], add_special_tokens=False)
        corpus['english_public_domain'] = english[8193:8193+8193]
    digest = hashlib.sha256(json.dumps(corpus, sort_keys=True).encode()).hexdigest()
    manifest = dict(corpus_sha256=digest, retrieval_seed=941 if args.heldout else 812, retrieval_cases=100,
                    gates=dict(max_relative_ppl_increase=.01, max_retrieval_drop=.02),
                    english_source='https://www.gutenberg.org/files/11/11-0.txt',
                    note='Chinese text is synthetic; code is local repository text. Not a broad quality benchmark.')
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, 'Frozen evaluation manifest changed'
    else:
        assert args.cache == 'auto', 'Create baseline manifest first'
        manifest_path.write_text(json.dumps(manifest, indent=2))
    model = llm.model_runner.model
    original_logits = model.compute_logits
    losses, top1, case_logits = [], [], []
    current_targets, current_start = None, 0

    @torch.inference_mode()
    def compute(hidden):
        nonlocal current_start
        if current_targets is not None:
            for offset in range(0, hidden.size(0), 64):
                h = hidden[offset:offset+64]
                logits = F.linear(h, model.lm_head.weight).float()
                target = torch.tensor(current_targets[current_start+offset:current_start+offset+h.size(0)],
                                      device=logits.device)
                losses.append(F.cross_entropy(logits, target, reduction='none').cpu())
                top1.append(logits.argmax(-1).cpu())
            current_start += hidden.size(0)
        logits = original_logits(hidden)
        if current_targets is None:
            case_logits.append(logits.detach().float().cpu())
        return logits

    model.compute_logits = compute

    def cold():
        llm.scheduler.block_manager = BlockManager(llm.config.num_kvcache_blocks, 256)

    scores = {}
    for name, ids in corpus.items():
        begin = len(losses)
        for start in range(0, len(ids)-1, 512):
            window = ids[start:start+513]
            if len(window) < 2:
                continue
            current_targets, current_start = window[1:], 0
            cold()
            llm.generate([window[:-1]], SamplingParams(max_tokens=1, ignore_eos=True), use_tqdm=False)
        domain_losses = torch.cat(losses[begin:])
        scores[name] = dict(tokens=domain_losses.numel(), nll=domain_losses.mean().item(),
                            ppl=math.exp(domain_losses.mean().item()))
        print(name, scores[name], flush=True)
    current_targets = None
    rng = random.Random(manifest['retrieval_seed'])
    answers, predictions, lengths = [], [], []
    candidate_ids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)]
    for i in range(100):
        answer = rng.randrange(10)
        room = rng.randrange(100, 999)
        fact = f'The secret digit for room {room} is {answer}.\n'
        filler = ('The garden has green trees. The library contains books. '
                  'The river flows past the village.\n') * (4, 12, 24)[i % 3]
        text = fact + filler if i % 2 else filler + fact
        text += f'\nWhat is the secret digit for room {room}? Answer with one digit only.'
        ids = tok.apply_chat_template([dict(role='user', content=text)], tokenize=True,
                                     add_generation_prompt=True, enable_thinking=False)
        cold()
        llm.generate([ids], SamplingParams(max_tokens=1, ignore_eos=True), use_tqdm=False)
        logits = case_logits[-1][0]
        pred = logits[candidate_ids].argmax().item()
        answers.append(answer)
        predictions.append(pred)
        lengths.append(len(ids))
        if (i+1) % 20 == 0:
            print(f'retrieval {i+1}/100', flush=True)
    tensors = dict(losses=torch.cat(losses), top1=torch.cat(top1),
                   retrieval_logits=torch.cat(case_logits))
    torch.save(tensors, root / f'{args.cache}.pt')
    report = dict(cache=args.cache, compiled=args.compile, cuda_graph=not llm.model_runner.enforce_eager, manifest=manifest, domains=scores,
                  overall_nll=tensors['losses'].mean().item(),
                  retrieval_accuracy=sum(a == b for a,b in zip(answers,predictions))/100,
                  retrieval_answers=answers, retrieval_predictions=predictions,
                  retrieval_context_lengths=lengths,
                  cache_data_bytes=llm.model_runner.cache_bytes,
                  cache_scale_bytes=llm.model_runner.scale_bytes,
                  kv_blocks=llm.config.num_kvcache_blocks)
    if args.cache != 'auto':
        ref = torch.load(root / 'auto.pt', weights_only=True)
        baseline = json.loads((root / 'auto.json').read_text())
        delta = tensors['retrieval_logits'] - ref['retrieval_logits']
        report.update(relative_ppl_change=math.exp(report['overall_nll']-baseline['overall_nll'])-1,
                      top1_agreement=(ref['top1'] == tensors['top1']).float().mean().item(),
                      retrieval_drop=baseline['retrieval_accuracy']-report['retrieval_accuracy'],
                      probe_max_abs_logit_error=delta.abs().max().item(),
                      probe_rms_logit_error=delta.square().mean().sqrt().item())
        report['quality_gates_passed'] = (report['relative_ppl_change'] <= .01
                                          and report['retrieval_drop'] <= .02)
    (root / f'{args.cache}.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if not k.startswith('retrieval_') or k == 'retrieval_accuracy'}, indent=2), flush=True)
    llm.exit()


if __name__ == '__main__':
    main()

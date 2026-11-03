"""Development ablation using FP32 fake-quant cache; never final validation.

Separates K and V error and compares established blockwise quantization sizes.
The old evaluation data is development data for this investigation. New heldout
inputs must be frozen before the selected storage format is validated.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from nanovllm import LLM, SamplingParams
from nanovllm.engine.block_manager import BlockManager
from nanovllm.layers.attention import Attention
from nanovllm.layers.kv_cache import quantize_reference


def main():
    llm = LLM(str(Path.home() / 'huggingface/Qwen3-0.6B'), attention_backend='triton',
              compile_model=False, enforce_eager=True, max_model_len=1024,
              max_num_batched_tokens=256, max_num_seqs=1, num_kvcache_blocks=4,
              gpu_memory_utilization=.85)
    runner, model, tok = llm.model_runner, llm.model_runner.model, llm.tokenizer
    runner.kv_cache = runner.kv_cache.float()
    attentions = [m for m in model.modules() if isinstance(m, Attention)]
    for i,m in enumerate(attentions):
        m.k_cache, m.v_cache = runner.kv_cache[:,i]
    tokens = tok.encode(Path('results/int8-eval/alice.txt').read_text(), add_special_tokens=False)[:2049]
    setting = (None, None)
    stats = []

    def transform(x, group):
        if group is None:
            return x.float()
        shaped = x.float().reshape(*x.shape[:-1], x.size(-1)//group, group)
        q,s = quantize_reference(shaped)
        return (q.float()*s.unsqueeze(-1)).reshape(x.shape)

    for layer,m in enumerate(attentions):
        original = m.forward
        def forward(q,k,v,original=original,layer=layer):
            kh, vh = transform(k, setting[0]), transform(v, setting[1])
            if setting == (128,128):
                stats.append(dict(layer=layer,
                                  k_mse=(kh-k.float()).square().mean().item(),
                                  v_mse=(vh-v.float()).square().mean().item(),
                                  k_max=k.abs().max().item(), v_max=v.abs().max().item()))
            return original(q,kh,vh)
        m.forward = forward
    original_logits = model.compute_logits
    targets, start, total, count = [], 0, 0., 0
    @torch.inference_mode()
    def compute(hidden):
        nonlocal start,total,count
        for i in range(0,hidden.size(0),64):
            h=hidden[i:i+64]
            logits=F.linear(h,model.lm_head.weight).float()
            y=torch.tensor(targets[start+i:start+i+h.size(0)],device=logits.device)
            total += F.cross_entropy(logits,y,reduction='sum').item()
            count += h.size(0)
        start += hidden.size(0)
        return original_logits(hidden)
    model.compute_logits=compute
    cases=[]
    for kg,vg in [(None,None),(128,None),(None,128),(128,128),(64,64),(32,32),(16,16),(32,128)]:
        setting=(kg,vg)
        total,count=0.,0
        for pos in range(0,2048,512):
            window=tokens[pos:pos+513]
            targets,start=window[1:],0
            llm.scheduler.block_manager=BlockManager(4,256)
            llm.generate([window[:-1]],SamplingParams(max_tokens=1,ignore_eos=True),use_tqdm=False)
        row=dict(k_group=kg,v_group=vg,tokens=count,nll=total/count)
        if cases:
            row['relative_ppl_change']=math.exp(row['nll']-cases[0]['nll'])-1
        cases.append(row)
        print(row,flush=True)
    Path('results/int8-ablation.json').write_text(json.dumps(dict(cases=cases,quantization_stats=stats,
        note='2048 English development tokens; FP32 fake-quant storage; not final INT8 quality evidence.'),indent=2))
    llm.exit()


if __name__=='__main__':
    main()

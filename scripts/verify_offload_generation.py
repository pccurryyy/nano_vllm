"""Unmodified sampler smoke test: mixed cache + mixed batching + both offloads."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nanovllm import LLM, SamplingParams


def main():
    llm = LLM(str(Path.home() / 'huggingface/Qwen3-0.6B'),
        attention_backend='triton', compile_model=False, enforce_eager=True,
        kv_cache_dtype='fp16_k_int8_v', enable_mixed_batch=True,
        enable_kv_offload=True, num_cpu_kvcache_blocks=4, num_kvcache_blocks=4,
        weight_offload_num_layers=2, weight_offload_window_size=1,
        fp16_reduced_precision_reduction=False,
        max_model_len=768, max_num_seqs=3, max_num_batched_tokens=64,
        gpu_memory_utilization=.7)
    try:
        prompts = [llm.tokenizer.apply_chat_template([dict(role='user', content=p)],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for p in ['1加1等于多少？', '用中文说一句问候。', '请输出数字3。']]
        outputs = llm.generate(prompts, SamplingParams(temperature=.00001, max_tokens=24), use_tqdm=False)
        assert len(outputs) == 3 and all(0 < len(x['token_ids']) <= 24 for x in outputs)
        assert llm.is_finished() and not llm.scheduler.block_manager.used_block_ids
        # Reuse the engine with identical input; no full-page hit is claimed
        # for these short prompts (prefix lifecycle is tested separately).
        reused = llm.generate([prompts[0]], SamplingParams(temperature=.00001, max_tokens=24), use_tqdm=False)
        assert 0 < len(reused[0]['token_ids']) <= 24
        assert llm.is_finished() and not llm.scheduler.block_manager.used_block_ids
        # Free generation across different batch shapes is not a bitwise
        # contract for this lossy format. Record equality rather than replacing
        # the fixed-history numerical tests with an invalid text-equality gate.
        report = dict(outputs=outputs, repeated_outputs=reused,
                      repeated_prompt_equal=reused[0]['token_ids'] == outputs[0]['token_ids'],
                      no_live_blocks=not llm.scheduler.block_manager.used_block_ids,
                      eos_observed=any(x['token_ids'][-1] == llm.tokenizer.eos_token_id for x in outputs),
                      request_metrics={str(i): llm.get_request_metrics(i) for i in llm.request_metrics},
                      note='Lifecycle/sampler smoke test, not a broad quality or cross-batch text-equality benchmark; pressure transfers tested separately')
        path = Path('results/offload-integration-matched/generation.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(json.dumps(report, indent=2, ensure_ascii=False))
    finally:
        llm.exit()


if __name__ == '__main__':
    main()

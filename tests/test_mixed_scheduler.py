import unittest
from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class MixedSchedulerTest(unittest.TestCase):
    def make_scheduler(self, budget=8):
        return Scheduler(SimpleNamespace(max_num_seqs=3, max_num_batched_tokens=budget,
            eos=99, kvcache_block_size=256, num_kvcache_blocks=8, enable_mixed_batch=True))

    def test_decode_and_partial_prefill_share_budget_without_early_output(self):
        s = self.make_scheduler()
        a = Sequence([10, 11], SamplingParams(max_tokens=3, ignore_eos=True))
        s.add(a)
        batch, phase = s.schedule()
        s.postprocess(batch, [42], phase)
        b = Sequence([12] * 12, SamplingParams(max_tokens=2, ignore_eos=True))
        s.add(b)
        batch, phase = s.schedule()
        self.assertEqual(batch, [a, b])
        self.assertTrue(phase)
        self.assertFalse(a.is_prefill)
        self.assertTrue(b.is_prefill)
        self.assertEqual([x.num_scheduled_tokens for x in batch], [1, 7])
        s.postprocess(batch, [42, 42], phase)
        self.assertEqual(b.num_completion_tokens, 0)
        self.assertEqual(b.num_cached_tokens, 7)
        for _ in range(8):
            if s.is_finished():
                break
            batch, phase = s.schedule()
            self.assertEqual(len(batch), len(set(x.seq_id for x in batch)))
            self.assertLessEqual(sum(x.num_scheduled_tokens for x in batch), 8)
            s.postprocess(batch, [42] * len(batch), phase)
        self.assertTrue(s.is_finished())
        self.assertEqual([a.num_completion_tokens, b.num_completion_tokens], [3, 2])
        self.assertEqual(len(s.block_manager.free_block_ids), 8)

    def test_eos_and_decode_budget_one(self):
        s = self.make_scheduler(budget=1)
        a = Sequence([10], SamplingParams(max_tokens=4))
        s.add(a)
        batch, phase = s.schedule()
        s.postprocess(batch, [42], phase)
        b = Sequence([11], SamplingParams(max_tokens=1))
        s.add(b)
        batch, phase = s.schedule()
        self.assertEqual(batch, [a])
        self.assertFalse(phase)
        s.postprocess(batch, [99], phase)
        batch, phase = s.schedule()
        self.assertEqual(batch, [b])
        s.postprocess(batch, [42], phase)
        self.assertTrue(s.is_finished())

    def test_offload_rejects_request_working_set_larger_than_gpu(self):
        s = Scheduler(SimpleNamespace(max_num_seqs=3, max_num_batched_tokens=256,
            eos=99, kvcache_block_size=256, num_kvcache_blocks=1,
            enable_mixed_batch=True, enable_kv_offload=True, num_cpu_kvcache_blocks=4))
        s.add(Sequence([10] * 257, SamplingParams(max_tokens=2)))
        with self.assertRaisesRegex(ValueError, "working set"):
            s.schedule()

    def test_decode_first_control_does_not_mix_prefill(self):
        s = Scheduler(SimpleNamespace(max_num_seqs=3, max_num_batched_tokens=8,
            eos=99, kvcache_block_size=256, num_kvcache_blocks=8,
            scheduling_policy="decode_first", enable_mixed_batch=False))
        a = Sequence([10], SamplingParams(max_tokens=2))
        s.add(a)
        batch, phase = s.schedule()
        s.postprocess(batch, [42], phase)
        b = Sequence([11] * 3, SamplingParams(max_tokens=1))
        s.add(b)
        batch, phase = s.schedule()
        self.assertEqual(batch, [a])
        self.assertFalse(phase)
        self.assertEqual(b.num_cached_tokens, 0)
        s.postprocess(batch, [42], phase)
        batch, phase = s.schedule()
        self.assertEqual(batch, [b])
        self.assertTrue(phase)

    def test_step_timestamp_metrics_are_not_client_latency(self):
        from nanovllm.engine.llm_engine import LLMEngine
        engine = LLMEngine.__new__(LLMEngine)
        engine.request_metrics = {1: dict(submitted=1., token_times=[2., 2.5, 3.], finished=3.),
                                  2: dict(submitted=1., token_times=[2.])}
        metric = engine.get_request_metrics(1)
        self.assertEqual(metric['ttft_seconds'], 1.)
        self.assertEqual(metric['tpot_seconds'], .5)
        self.assertEqual(metric['latency_seconds'], 2.)
        self.assertIsNone(engine.get_request_metrics(2)['tpot_seconds'])
        self.assertEqual(metric['timestamp_scope'], 'engine_step_completion')


if __name__ == "__main__":
    unittest.main()

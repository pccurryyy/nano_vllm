import unittest
from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class SchedulerTest(unittest.TestCase):
    def test_preemption_and_resume_without_leaking_pages(self):
        scheduler = Scheduler(SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=512,
                                             eos=-1, kvcache_block_size=256, num_kvcache_blocks=2))
        params = SamplingParams(max_tokens=3, ignore_eos=True)
        seqs = [Sequence([i + 10] * 256, params) for i in range(2)]
        for seq in seqs:
            scheduler.add(seq)
        scheduled, prefill = scheduler.schedule()
        self.assertEqual(len(scheduled), 2)
        scheduler.postprocess(scheduled, [1, 2], prefill)
        scheduled, prefill = scheduler.schedule()
        self.assertFalse(prefill)
        self.assertEqual(len(scheduled), 1)
        self.assertEqual(len(scheduler.waiting), 1)  # Other sequence was preempted.
        scheduler.postprocess(scheduled, [3], prefill)
        steps = 0
        while not scheduler.is_finished():
            scheduled, prefill = scheduler.schedule()
            scheduler.postprocess(scheduled, [4] * len(scheduled), prefill)
            steps += 1
            self.assertLess(steps, 10)
        self.assertEqual([s.num_completion_tokens for s in seqs], [3, 3])
        self.assertEqual(len(scheduler.block_manager.free_block_ids), 2)
        self.assertEqual(scheduler.block_manager.used_block_ids, set())

import unittest
from types import SimpleNamespace
import torch

from nanovllm.engine.kv_offload import BlockTransfer, TieredBlockManager
from nanovllm.engine.sequence import Sequence


class KVOffloadTest(unittest.TestCase):
    device = "cpu"

    def setUp(self):
        self.old_block_size = Sequence.block_size
        Sequence.block_size = 4
        self.bm = TieredBlockManager(2, 4, 2)
        # Different dtypes, including both K/V scale planes for full INT8.
        self.pools = [torch.zeros(2, 2, 4, 1, 3, dtype=d, device=self.device)
                      for d in (torch.float16, torch.int8)]
        self.pools += [torch.zeros(2, 2, 4, 1, dtype=torch.float32, device=self.device)]
        self.transfer = BlockTransfer(self.pools, 2)
        self.bm.bind(self.transfer)
        self.seqs = [Sequence([i + 10] * 4) for i in range(4)]
        for seq in self.seqs:
            self.bm.allocate(seq, 0)

    def tearDown(self):
        Sequence.block_size = self.old_block_size

    def fill(self, seq, value):
        page = self.bm.gpu_table(seq)[0]
        for pool in self.pools:
            pool[:, page].fill_(value)

    def check(self, seq, value):
        page = self.bm.gpu_table(seq)[0]
        for pool in self.pools:
            torch.testing.assert_close(pool[:, page], torch.full_like(pool[:, page], value), rtol=0, atol=0)

    def test_full_pools_swap_and_rewrite_never_reads_stale_scale(self):
        a, b, c, d = self.seqs
        self.bm.ensure_resident([a, b])
        self.fill(a, 11); self.fill(b, 22)
        self.bm.ensure_resident([c, d])
        self.fill(c, 33); self.fill(d, 44)
        self.assertEqual(len(self.bm.free_cpu), 0)
        self.bm.ensure_resident([a, b])  # Requires CPU scratch swap, no free slots.
        self.check(a, 11); self.check(b, 22)
        self.fill(a, 55)  # New GPU write must replace, not reuse, the old CPU copy.
        self.bm.ensure_resident([c, d])
        self.check(c, 33); self.check(d, 44)
        self.bm.ensure_resident([a])
        self.check(a, 55)
        self.assertGreater(self.transfer.stats["h2d"], 0)
        for seq in self.seqs:
            self.bm.deallocate(seq)
        self.assertEqual(len(self.bm.free_block_ids), 4)
        self.assertFalse(self.bm.used_block_ids)

    def test_protected_union_and_oversized_working_set(self):
        a, b, c, _ = self.seqs
        self.bm.ensure_resident([a, b])
        self.fill(a, 7); self.fill(b, 9)
        self.bm.ensure_resident([a, c])
        self.check(a, 7)
        with self.assertRaisesRegex(ValueError, "working set"):
            self.bm.ensure_resident([a, b, c])

    def test_lru_and_random_round_trips_with_gpu_rewrites(self):
        import random
        a, b, c, _ = self.seqs
        self.bm.ensure_resident([a, b]); self.fill(a, 1); self.fill(b, 2)
        self.bm.ensure_resident([a])
        self.bm.ensure_resident([c])
        self.assertIsNotNone(self.bm.blocks[b.block_table[0]].cpu_id)
        self.fill(c, 3)
        # Initialize D before checking the complete set of four logical pages.
        d = self.seqs[3]
        self.bm.ensure_resident([d]); self.fill(d, 4)
        expected = [1, 2, 3, 4]
        rng = random.Random(17)
        for step in range(16):
            indices = rng.sample(range(4), 2)
            self.bm.ensure_resident([self.seqs[i] for i in indices])
            for i in indices:
                self.check(self.seqs[i], expected[i])
            i = indices[0]
            expected[i] = step + 20
            self.fill(self.seqs[i], expected[i])
            gpu_ids = [b.gpu_id for b in self.bm.blocks if b.gpu_id is not None]
            cpu_ids = [b.cpu_id for b in self.bm.blocks if b.cpu_id is not None]
            self.assertEqual(len(gpu_ids), len(set(gpu_ids)))
            self.assertEqual(len(cpu_ids), len(set(cpu_ids)))

    def test_shared_reference_and_idle_reclamation(self):
        a, b, c, _ = self.seqs
        shared = SimpleNamespace(block_table=list(a.block_table), num_cached_tokens=4)
        self.bm.blocks[a.block_table[0]].ref_count += 1
        self.bm.ensure_resident([a, b]); self.fill(a, 3)
        self.bm.deallocate(a)
        self.bm.ensure_resident([shared, c])
        self.check(shared, 3)
        self.bm.deallocate(shared)
        self.assertEqual(self.bm.blocks[0].ref_count, 0)

    def test_prefix_hash_survives_migration_and_idle_discard_invalidates_it(self):
        for seq in self.seqs:
            self.bm.deallocate(seq)
        a = Sequence([10] * 8)
        self.bm.allocate(a, 0)
        self.bm.ensure_resident([a])
        self.fill(a, 17)
        a.num_scheduled_tokens = 8
        self.bm.hash_blocks(a)
        b = Sequence([10] * 4 + [20] * 4)
        self.assertEqual(self.bm.can_allocate(b), 1)
        self.bm.allocate(b, 1)
        shared_id = a.block_table[0]
        self.assertEqual(b.block_table[0], shared_id)
        self.bm.ensure_resident([b])
        self.check(b, 17)
        self.bm.deallocate(a)
        self.assertEqual(self.bm.blocks[shared_id].ref_count, 1)
        self.bm.deallocate(b)
        block = self.bm.blocks[shared_id]
        old_hash = block.hash
        self.bm._discard_idle(block)
        self.assertNotIn(old_hash, self.bm.hash_to_block_id)
        self.assertIsNone(block.cpu_id)
        self.assertIsNone(block.gpu_id)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class GPUKVOffloadTest(KVOffloadTest):
    device = "cuda"


if __name__ == "__main__":
    unittest.main()

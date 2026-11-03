from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.mixed = getattr(config, "enable_mixed_batch", False)
        self.offload = getattr(config, "enable_kv_offload", False)
        self.decode_first = self.mixed or getattr(config, "scheduling_policy", "prefill_first") == "decode_first"
        if self.offload:
            from nanovllm.engine.kv_offload import TieredBlockManager
            self.block_manager = TieredBlockManager(config.num_kvcache_blocks,
                config.kvcache_block_size, config.num_cpu_kvcache_blocks)
        self.preemptions = 0
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        if self.mixed or self.offload or self.decode_first:
            return self._schedule_extended()
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        self.preemptions += 1
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def _schedule_extended(self):
        """Decode-first mixed batches; offload-only retains prefill priority.

        Each selected sequence declares its own semantic phase. The bool return
        chooses a variable-length implementation, not every sequence's phase.
        GPU residency of the union is checked before committing a selection.
        """
        bm = self.block_manager
        limit = getattr(bm, "num_gpu_blocks", len(bm.blocks))
        selected, protected = [], set()
        tokens = 0

        def prefill():
            nonlocal tokens
            while self.waiting and len(selected) < self.max_num_seqs and tokens < self.max_num_batched_tokens:
                seq = self.waiting[0]
                if seq.num_blocks > limit:
                    raise ValueError("A request's full KV working set exceeds GPU block capacity")
                # Conservatively reserve the complete request, including blocks
                # not yet computed. Offload is across requests, not tile streaming.
                extra = len(set(seq.block_table) - protected) if seq.block_table else seq.num_blocks
                if len(protected) + extra > limit:
                    break
                if not seq.block_table:
                    cached = bm.can_allocate(seq)
                    if cached < 0:
                        break
                    bm.allocate(seq, cached)
                seq.is_prefill = True
                seq.num_scheduled_tokens = min(seq.num_tokens - seq.num_cached_tokens,
                                               self.max_num_batched_tokens - tokens)
                tokens += seq.num_scheduled_tokens
                selected.append(seq)
                protected.update(seq.block_table)
                if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                    self.waiting.popleft()
                    seq.status = SequenceStatus.RUNNING
                    self.running.append(seq)

        def decode():
            nonlocal tokens
            # Snapshot excludes newly admitted prefills and prevents selecting
            # any request twice in the same iteration.
            for seq in list(self.running):
                if seq in selected or seq not in self.running:
                    continue
                if len(selected) >= self.max_num_seqs or tokens >= self.max_num_batched_tokens:
                    break
                if seq.num_blocks > limit:
                    raise ValueError("A decode request's full KV working set exceeds GPU block capacity")
                new_page = int(len(seq) % self.block_size == 1)
                if len(protected | set(seq.block_table)) + new_page > limit:
                    continue
                while not bm.can_append(seq):
                    victims = [s for s in self.running if s is not seq and s not in selected]
                    if not victims or self.offload:
                        break
                    victim = victims[-1]
                    self.running.remove(victim)
                    self.preempt(victim)
                if not bm.can_append(seq):
                    continue
                bm.may_append(seq)
                seq.is_prefill = False
                seq.num_scheduled_tokens = 1
                selected.append(seq)
                protected.update(seq.block_table)
                tokens += 1
                # Rotate so a GPU-sized subset does not monopolize the pool.
                self.running.remove(seq)
                self.running.append(seq)

        if self.decode_first:
            decode()
            if self.mixed or not selected:
                prefill()
        else:
            prefill()
            if not selected:
                decode()
        if not selected:
            raise RuntimeError("Scheduler made no progress: KV capacity or backing store exhausted")
        return selected, any(seq.is_prefill for seq in selected)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

"""Synchronous, format-aware KV paging between GPU and host.

Design reference: HuangRZzzz/nano-vllm-GPU-offload, GPU_OFFLOAD branch,
6864dede7f51d9cf149f11d0a6e5928994c2a03e (MIT). Independently integrated
with this checkout's allocator and heterogeneous K/V/scale pools.

One authoritative copy per logical block: reloading consumes the CPU copy.
Thus subsequent GPU writes cannot leave a stale CPU replica. A host scratch
block permits swaps even when both pools are completely occupied.
"""
from collections import deque
from time import perf_counter
import torch

from nanovllm.engine.block_manager import BlockManager


class BlockTransfer:
    def __init__(self, pools, cpu_blocks):
        self.gpu = [pool for pool in pools if pool is not None]
        self.cpu = [torch.empty((p.size(0), cpu_blocks, *p.shape[2:]),
                    dtype=p.dtype, device="cpu", pin_memory=p.is_cuda) for p in self.gpu]
        self.scratch = [torch.empty((p.size(0), *p.shape[2:]), dtype=p.dtype,
                        device="cpu", pin_memory=p.is_cuda) for p in self.gpu]
        self.block_bytes = sum(p[:, 0].numel() * p.element_size() for p in self.gpu)
        self.cpu_bytes = sum(p.numel() * p.element_size() for p in self.cpu)
        self.scratch_bytes = sum(p.numel() * p.element_size() for p in self.scratch)
        self.stats = dict(d2h=0, h2d=0, d2h_bytes=0, h2d_bytes=0, seconds=0.)

    def copy(self, gpu_id, cpu_id, to_cpu):
        start = perf_counter()
        for gpu, cpu in zip(self.gpu, self.cpu):
            src, dst = (gpu[:, gpu_id], cpu[:, cpu_id]) if to_cpu else (cpu[:, cpu_id], gpu[:, gpu_id])
            dst.copy_(src, non_blocking=False)
        direction = "d2h" if to_cpu else "h2d"
        self.stats[direction] += 1
        self.stats[direction + "_bytes"] += self.block_bytes
        self.stats["seconds"] += perf_counter() - start

    def swap(self, gpu_id, cpu_id):
        start = perf_counter()
        for gpu, cpu, scratch in zip(self.gpu, self.cpu, self.scratch):
            scratch.copy_(gpu[:, gpu_id], non_blocking=False)
            gpu[:, gpu_id].copy_(cpu[:, cpu_id], non_blocking=False)
            cpu[:, cpu_id].copy_(scratch)
        for direction in ("d2h", "h2d"):
            self.stats[direction] += 1
            self.stats[direction + "_bytes"] += self.block_bytes
        self.stats["seconds"] += perf_counter() - start


class TieredBlockManager(BlockManager):
    def __init__(self, num_gpu_blocks, block_size, num_cpu_blocks):
        super().__init__(num_gpu_blocks + num_cpu_blocks, block_size)
        self.num_gpu_blocks = num_gpu_blocks
        self.free_gpu = deque(range(num_gpu_blocks))
        self.free_cpu = deque(range(num_cpu_blocks))
        self.transfer = None
        self.clock = 0
        for block in self.blocks:
            block.gpu_id = block.cpu_id = None
            block.last_access = 0

    def bind(self, transfer):
        self.transfer = transfer

    def _release_storage(self, block):
        if block.gpu_id is not None:
            self.free_gpu.append(block.gpu_id)
        if block.cpu_id is not None:
            self.free_cpu.append(block.cpu_id)
        block.gpu_id = block.cpu_id = None

    def _allocate_block(self):
        # The original allocator retains idle prefixes. Reusing their logical
        # identity must first release either physical location and old mapping.
        block = self.blocks[self.free_block_ids[0]]
        self._release_storage(block)
        return super()._allocate_block()

    def _discard_idle(self, block):
        assert block.ref_count == 0
        if self.hash_to_block_id.get(block.hash) == block.block_id:
            del self.hash_to_block_id[block.hash]
        block.hash = -1
        block.token_ids = []
        self._release_storage(block)

    def ensure_resident(self, seqs):
        if self.transfer is None:
            raise RuntimeError("KV transfer pools are not bound")
        ids = list(dict.fromkeys(i for seq in seqs for i in seq.block_table))
        protected = set(ids)
        if len(protected) > self.num_gpu_blocks:
            raise ValueError("Active KV working set exceeds GPU blocks; offload is across requests")
        for logical_id in ids:
            block = self.blocks[logical_id]
            if block.gpu_id is None:
                if not self.free_gpu:
                    candidates = [b for b in self.blocks
                                  if b.gpu_id is not None and b.block_id not in protected]
                    if not candidates:
                        raise RuntimeError("No unprotected GPU block can be evicted")
                    victim = min(candidates, key=lambda b: (b.ref_count != 0, b.last_access))
                    if victim.ref_count == 0:
                        self._discard_idle(victim)
                    else:
                        if not self.free_cpu:
                            idle_cpu = [b for b in self.blocks if b.cpu_id is not None and b.ref_count == 0]
                            if idle_cpu:
                                self._discard_idle(min(idle_cpu, key=lambda b: b.last_access))
                        if self.free_cpu:
                            cpu_id = self.free_cpu.popleft()
                            self.transfer.copy(victim.gpu_id, cpu_id, to_cpu=True)
                            victim.cpu_id = cpu_id
                            self.free_gpu.append(victim.gpu_id)
                            victim.gpu_id = None
                        elif block.cpu_id is not None:
                            # Full host pool: swap through one host scratch
                            # block rather than allocate extra GPU memory.
                            gpu_id, cpu_id = victim.gpu_id, block.cpu_id
                            self.transfer.swap(gpu_id, cpu_id)
                            victim.gpu_id, victim.cpu_id = None, cpu_id
                            block.gpu_id, block.cpu_id = gpu_id, None
                        else:
                            raise RuntimeError("CPU KV backing store exhausted")
                if block.gpu_id is None:
                    gpu_id = self.free_gpu.popleft()
                    if block.cpu_id is not None:
                        self.transfer.copy(gpu_id, block.cpu_id, to_cpu=False)
                        self.free_cpu.append(block.cpu_id)
                        block.cpu_id = None
                    block.gpu_id = gpu_id
            self.clock += 1
            block.last_access = self.clock

    def gpu_table(self, seq):
        result = [self.blocks[i].gpu_id for i in seq.block_table]
        if any(i is None for i in result):
            raise RuntimeError("KV block must be resident before building GPU metadata")
        return result

    def deallocate(self, seq):
        # Preserve unreferenced prefixes, as the original allocator does.
        # Free physical slots can be reclaimed on demand by _discard_idle.
        super().deallocate(seq)

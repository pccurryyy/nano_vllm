import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.layers.kv_cache import INT8_CACHE, MIXED_CACHE, cache_block_bytes
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.layers.attention import Attention, flash_attn_varlen_func
from nanovllm.utils.compile import set_compile_enabled


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.block_manager = None

        torch.cuda.set_device(rank)
        if config.fp16_reduced_precision_reduction is not None:
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = config.fp16_reduced_precision_reduction
        if torch.cuda.get_device_capability(rank)[0] < 8:
            hf_config.dtype = torch.float16
        if config.kv_cache_dtype == MIXED_CACHE and hf_config.dtype != torch.float16:
            raise ValueError("fp16_k_int8_v requires an FP16 execution dtype")
        backend = config.attention_backend
        if backend == "auto":
            backend = ("flash" if config.kv_cache_dtype == "auto" and flash_attn_varlen_func is not None
                       and torch.cuda.get_device_capability(rank)[0] >= 8 else "triton")
        if backend == "flash" and (flash_attn_varlen_func is None or torch.cuda.get_device_capability(rank)[0] < 8):
            raise RuntimeError("Standard flash-attn requires a supported GPU (SM80+) and an installed flash-attn package. Use attention_backend='triton' on this GTX 1650.")
        if backend == "torch" and not self.enforce_eager:
            raise ValueError("The torch reference backend requires enforce_eager=True; select triton for CUDA Graphs.")
        self.attention_backend = backend
        self.compile_model = config.compile_model if config.compile_model is not None else backend != "torch"
        set_compile_enabled(self.compile_model)
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cpu" if config.weight_offload_num_layers else "cuda")
        try:
            self.model = Qwen3ForCausalLM(hf_config)
            for module in self.model.modules():
                if isinstance(module, Attention):
                    module.backend = backend
                    module.use_flash = backend == "flash"
                    module.kv_cache_dtype = config.kv_cache_dtype
            load_model(self.model, config.model)
            self.model.eval()
            if config.weight_offload_num_layers:
                from nanovllm.engine.weight_offload import WeightOffloadManager
                from nanovllm.layers.rotary_embedding import RotaryEmbedding
                for module in self.model.modules():
                    if isinstance(module, RotaryEmbedding):
                        module.rebuild_cache("cuda")
                base = self.model.model
                base.embed_tokens.to("cuda")
                base.norm.to("cuda")
                self.model.lm_head.to("cuda")
                if hf_config.tie_word_embeddings:
                    self.model.lm_head.weight.data = base.embed_tokens.weight.data
                self.weight_offload = WeightOffloadManager(base.layers,
                    config.weight_offload_num_layers, config.weight_offload_window_size)
                base.weight_offload_manager = self.weight_offload
            torch.set_default_device("cuda")
            self.sampler = Sampler()
            self.warmup_model()
            self.allocate_kv_cache()
            if not self.enforce_eager:
                self.capture_cudagraph()
        except Exception:
            reset_context()
            dist.destroy_process_group()
            raise
        finally:
            torch.set_default_device("cpu")
            torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def bind_block_manager(self, block_manager):
        self.block_manager = block_manager
        if self.config.enable_kv_offload:
            from nanovllm.engine.kv_offload import BlockTransfer
            self.kv_transfer = BlockTransfer([*self.kv_cache, *self.kv_scales],
                                             self.config.num_cpu_kvcache_blocks)
            block_manager.bind(self.kv_transfer)

    def _gpu_table(self, seq):
        if self.config.enable_kv_offload and self.block_manager is not None:
            return self.block_manager.gpu_table(seq)
        return seq.block_table

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        self.workspace_bytes = max(0, torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated())
        if self.config.kv_cache_dtype != "auto":
            # Warm the real quantized paged path before budgeting the final pool.
            pages_per_seq = (seq_len + self.block_size - 1) // self.block_size
            layers = self.config.hf_config.num_hidden_layers
            heads = self.config.hf_config.num_key_value_heads
            dim = getattr(self.config.hf_config, "head_dim",
                          self.config.hf_config.hidden_size // self.config.hf_config.num_attention_heads)
            self._make_cache(layers, num_seqs * pages_per_seq, heads, dim)
            for i, seq in enumerate(seqs):
                seq.block_table = list(range(i * pages_per_seq, (i + 1) * pages_per_seq))
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            self.run(seqs, True)
            self.workspace_bytes = max(self.workspace_bytes, torch.cuda.max_memory_allocated() - before)
            for module in self.model.modules():
                if isinstance(module, Attention):
                    module.k_cache = module.v_cache = torch.tensor([])
                    module.k_scale_cache = module.v_scale_cache = None
            del self.kv_cache, self.kv_scales
        torch.cuda.empty_cache()

    def _make_cache(self, layers, blocks, heads, dim):
        fmt = self.config.kv_cache_dtype
        shape = (layers, blocks, self.block_size, heads, dim)
        if fmt == MIXED_CACHE:
            self.kv_cache = (torch.empty(shape, dtype=torch.float16),
                             torch.empty(shape, dtype=torch.int8))
            self.kv_scales = (None, torch.empty(shape[:-1], dtype=torch.float32))
        else:
            self.kv_cache = torch.empty((2, *shape),
                dtype=torch.int8 if fmt == INT8_CACHE else self.config.hf_config.dtype)
            self.kv_scales = (torch.empty((2, *shape[:-1]), dtype=torch.float32)
                              if fmt == INT8_CACHE else (None, None))
        layer_id = 0
        for module in self.model.modules():
            if isinstance(module, Attention):
                module.k_cache, module.v_cache = (pool[layer_id] for pool in self.kv_cache)
                module.k_scale_cache, module.v_scale_cache = (
                    pool[layer_id] if pool is not None else None for pool in self.kv_scales)
                layer_id += 1

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = cache_block_bytes(hf_config.num_hidden_layers, self.block_size,
                                       num_kv_heads, head_dim, hf_config.dtype,
                                       config.kv_cache_dtype == INT8_CACHE,
                                       config.kv_cache_dtype == MIXED_CACHE)
        available_blocks = int(total * config.gpu_memory_utilization - used - self.workspace_bytes) // block_bytes
        if config.num_kvcache_blocks > 0:
            if config.num_kvcache_blocks > available_blocks:
                raise ValueError("Requested KV block count exceeds measured memory budget")
        else:
            config.num_kvcache_blocks = available_blocks
        if config.num_kvcache_blocks <= 0:
            raise ValueError("No KV blocks fit in the measured memory budget")
        self._make_cache(hf_config.num_hidden_layers, config.num_kvcache_blocks, num_kv_heads, head_dim)
        self.cache_bytes = sum(pool.numel() * pool.element_size() for pool in self.kv_cache)
        self.scale_bytes = sum(pool.numel() * pool.element_size() for pool in self.kv_scales if pool is not None)

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [self._gpu_table(seq) + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            gpu_table = self._gpu_table(seq)
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = gpu_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = gpu_table[i] * self.block_size + self.block_size
                else:
                    slot_end = gpu_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1] or (
            self.config.kv_cache_dtype != "auto" and all(seq.block_table for seq in seqs)
        ):    # prefix cache, or consistent quantization for all production prefill
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(self._gpu_table(seq)[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            if context.block_tables.size(1) > self.graph_vars["block_tables"].size(1):
                raise ValueError("Request block table exceeds captured graph capacity")
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if self.config.enable_kv_offload and self.block_manager is not None:
            # Synchronous copies finish outside the graph; captured pool bases
            # stay fixed. Protect the complete union for this model invocation.
            self.block_manager.ensure_resident(seqs)
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        # Capture's zero-length rows are padding, not writers to slot zero.
        slot_mapping = torch.full((max_bs,), -1, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = sorted(set(bs for bs in [1, 2, 4, 8] + list(range(16, max_bs + 1, 16)) + [max_bs] if bs <= max_bs))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

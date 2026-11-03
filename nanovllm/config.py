import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    attention_backend: str = "auto"
    compile_model: bool | None = None
    kv_cache_dtype: str = "auto"
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    enable_mixed_batch: bool = False
    enable_kv_offload: bool = False
    num_cpu_kvcache_blocks: int = 0
    weight_offload_num_layers: int = 0
    weight_offload_window_size: int = 1
    fp16_reduced_precision_reduction: bool | None = None
    scheduling_policy: str = "prefill_first"

    def __post_init__(self):
        if self.scheduling_policy not in ("prefill_first", "decode_first"):
            raise ValueError("scheduling_policy must be prefill_first or decode_first")
        if self.num_cpu_kvcache_blocks < 0 or self.weight_offload_num_layers < 0 or self.weight_offload_window_size < 1:
            raise ValueError("Offload counts must be nonnegative and the weight window positive")
        if self.enable_kv_offload and self.num_cpu_kvcache_blocks < 1:
            raise ValueError("KV offload requires a positive num_cpu_kvcache_blocks")
        if (self.enable_mixed_batch or self.enable_kv_offload or self.weight_offload_num_layers or self.scheduling_policy != "prefill_first") and self.tensor_parallel_size != 1:
            raise ValueError("Mixed batching and offload currently support TP=1 only")
        if self.weight_offload_num_layers:
            if self.compile_model is True:
                raise ValueError("Weight offload currently requires compile_model=False")
            self.compile_model = False
            self.enforce_eager = True
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if min(self.max_num_batched_tokens, self.max_num_seqs, self.max_model_len, self.kvcache_block_size) < 1:
            raise ValueError("Batch, length and block limits must be positive")
        if self.num_kvcache_blocks != -1 and self.num_kvcache_blocks < 1:
            raise ValueError("num_kvcache_blocks must be -1 (automatic) or positive")
        if self.kv_cache_dtype not in ("auto", "int8_per_token_head", "fp16_k_int8_v"):
            raise ValueError("kv_cache_dtype must be auto, int8_per_token_head or fp16_k_int8_v")
        if self.kv_cache_dtype != "auto" and (self.tensor_parallel_size != 1 or self.attention_backend == "flash"):
            raise ValueError("INT8 KV supports single-GPU triton or eager torch, not flash or tensor parallelism")
        if self.attention_backend not in ("auto", "torch", "triton", "flash"):
            raise ValueError("attention_backend must be auto, torch, triton, or flash")
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        if self.weight_offload_num_layers and self.weight_offload_num_layers > self.hf_config.num_hidden_layers:
            raise ValueError("weight_offload_num_layers exceeds model layer count")
        head_dim = getattr(self.hf_config, "head_dim", self.hf_config.hidden_size // self.hf_config.num_attention_heads)
        if self.kv_cache_dtype != "auto" and head_dim != 128:
            raise ValueError("INT8 KV is currently validated for head_dim=128")
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)

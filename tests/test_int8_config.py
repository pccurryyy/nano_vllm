import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nanovllm.config import Config
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams


class ConfigAndAdmissionTest(unittest.TestCase):
    def test_offload_configuration_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            for kwargs in [dict(enable_kv_offload=True),
                           dict(scheduling_policy='invalid'),
                           dict(scheduling_policy='decode_first', tensor_parallel_size=2),
                           dict(num_cpu_kvcache_blocks=-1),
                           dict(weight_offload_num_layers=-1),
                           dict(weight_offload_window_size=0),
                           dict(weight_offload_num_layers=1, compile_model=True),
                           dict(enable_mixed_batch=True, tensor_parallel_size=2),
                           dict(enable_kv_offload=True, num_cpu_kvcache_blocks=2, tensor_parallel_size=2)]:
                with self.assertRaises(ValueError):
                    Config(directory, **kwargs)
            hf = SimpleNamespace(head_dim=128, hidden_size=1024, num_attention_heads=16,
                                 max_position_embeddings=4096, num_hidden_layers=28)
            with patch('nanovllm.config.AutoConfig.from_pretrained', return_value=hf):
                config = Config(directory, weight_offload_num_layers=2)
                self.assertTrue(config.enforce_eager)
                self.assertFalse(config.compile_model)
                with self.assertRaisesRegex(ValueError, 'layer count'):
                    Config(directory, weight_offload_num_layers=29)

    def test_unsupported_modes_and_dimension(self):
        with tempfile.TemporaryDirectory() as directory:
            for kwargs in [dict(kv_cache_dtype='bad'),
                           dict(kv_cache_dtype='fp16_k_int8_v', attention_backend='flash'),
                           dict(kv_cache_dtype='fp16_k_int8_v', tensor_parallel_size=2),
                           dict(kv_cache_dtype='int8_per_token_head', attention_backend='flash'),
                           dict(kv_cache_dtype='int8_per_token_head', tensor_parallel_size=2)]:
                with self.assertRaises(ValueError):
                    Config(directory, **kwargs)
            with patch('nanovllm.config.AutoConfig.from_pretrained',
                       return_value=SimpleNamespace(head_dim=64, hidden_size=1024,
                                                    num_attention_heads=16, max_position_embeddings=4096)):
                with self.assertRaises(ValueError):
                    Config(directory, kv_cache_dtype='int8_per_token_head')

    def test_request_length_boundary_and_closed_engine(self):
        engine = LLMEngine.__new__(LLMEngine)
        engine.config = SimpleNamespace(max_model_len=10)
        accepted = []
        engine.scheduler = SimpleNamespace(add=accepted.append)
        engine.add_request([1]*8, SamplingParams(max_tokens=2))
        self.assertEqual(len(accepted), 1)
        with self.assertRaises(ValueError):
            engine.add_request([1]*9, SamplingParams(max_tokens=2))
        with self.assertRaises(ValueError):
            engine.add_request([], SamplingParams(max_tokens=2))
        with self.assertRaises(ValueError):
            engine.add_request([1], SamplingParams(max_tokens=0))
        engine.exit()  # Explicit cleanup followed by atexit must be harmless.

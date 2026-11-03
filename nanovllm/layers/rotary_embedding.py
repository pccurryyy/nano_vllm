from functools import lru_cache
import torch
from torch import nn
from nanovllm.utils.compile import compile_if_supported


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.register_buffer("cos_sin_cache", self._build_cache(torch.get_default_device()), persistent=False)

    def _build_cache(self, device):
        inv_freq = 1.0 / (self.base**(torch.arange(0, self.rotary_dim, 2, dtype=torch.float, device=device) / self.rotary_dim))
        t = torch.arange(self.max_position_embeddings, dtype=torch.float, device=device)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        return torch.cat((cos, sin), dim=-1).unsqueeze_(1)

    def rebuild_cache(self, device):
        # CPU model construction must not change the arithmetic used to create
        # execution-time RoPE tables. A CPU-computed trig table merely moved to
        # CUDA can differ from the GPU-created baseline and amplify INT8 rounds.
        self.cos_sin_cache = self._build_cache(device)

    @compile_if_supported
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb

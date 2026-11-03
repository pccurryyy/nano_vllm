import torch
from functools import wraps

_enabled = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def set_compile_enabled(enabled):
    global _enabled
    _enabled = enabled


def compile_if_supported(fn):
    """Allow explicitly tested older Triton versions to compile on Turing."""
    compiled = torch.compile(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        return (compiled if _enabled else fn)(*args, **kwargs)
    return wrapper

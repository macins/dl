from __future__ import annotations

from .registry import register_block
from ...utils.externals import ensure_torch

torch = ensure_torch()
nn = torch.nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = x_float * rms
        return out.to(dtype=x.dtype) * self.weight


def build_norm(name: str, dim: int, eps: float = 1e-6) -> nn.Module:
    key = str(name).strip().lower()
    if key == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    if key == "rmsnorm":
        return RMSNorm(dim, eps=eps)
    raise KeyError(f"Unknown norm: {name}")

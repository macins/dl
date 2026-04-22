from __future__ import annotations

from ..base import BaseHead
from .registry import register_head
from ...utils.externals import ensure_torch

torch = ensure_torch()
nn = torch.nn


@register_head("sequence_regression")
class SequenceRegressionHead(BaseHead):
    def __init__(
        self,
        *,
        input_dim: int,
        target_key: str = "ret_30min",
        output_dim: int = 1,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.target_key = str(target_key)
        self.norm = nn.LayerNorm(input_dim) if use_layer_norm else nn.Identity()
        self.proj = nn.Linear(input_dim, int(output_dim))

    def forward(self, x: torch.Tensor) -> dict:
        pred = self.proj(self.norm(x))
        return {"preds": {self.target_key: pred}}

from __future__ import annotations

from ..base import BaseHead
from .registry import register_head
from ...utils.externals import ensure_torch

torch = ensure_torch()
nn = torch.nn


@register_head("sequence_regression")
class SequenceRegressionHead(BaseHead):
    """
    Supports arbitrary leading dimensions.

    Input:
        x: (..., input_dim)

    Output:
        pred: (..., output_dim)

    Examples:
        (B, T, D)    -> (B, T, 1)
        (B, N, T, D) -> (B, N, T, 1)
    """

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
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

        self.norm = nn.LayerNorm(self.input_dim) if use_layer_norm else nn.Identity()
        self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> dict:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"SequenceRegressionHead expected last dim={self.input_dim}, "
                f"got {x.shape[-1]} for input shape {tuple(x.shape)}."
            )

        pred = self.proj(self.norm(x))
        return {"preds": {self.target_key: pred}}
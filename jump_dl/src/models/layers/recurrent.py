from __future__ import annotations

from .registry import register_block
from ...utils.externals import ensure_torch

torch = ensure_torch()
nn = torch.nn


@register_block("gru_residual")
class ResidualGRUBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity()
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            dropout=0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h, _ = self.gru(self.norm(x))
        out = self.dropout(h) + x
        if padding_mask is not None:
            out = out * padding_mask.unsqueeze(-1).to(dtype=out.dtype)
        return out

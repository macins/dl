from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..base import BaseBackbone
from ..layers import build_block
from .registry import register_backbone
from ...utils.externals import ensure_torch

torch = ensure_torch()


@register_backbone("stacked_blocks")
@register_backbone("gru_sequence")
class GRUSequenceBackbone(BaseBackbone):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        blocks: Sequence[Mapping[str, object]] | None = None,
    ) -> None:
        super().__init__()
        if blocks is None:
            blocks = [
                {"name": "gru_residual", "dropout": dropout}
                for _ in range(int(num_layers))
            ]
        self.blocks = torch.nn.ModuleList(
            [build_block(block_cfg, hidden_size=int(hidden_size)) for block_cfg in blocks]
        )
        self.output_dim = int(hidden_size)

    def forward(self, x, *, padding_mask=None):
        for block in self.blocks:
            x = block(x, padding_mask=padding_mask)
        return x

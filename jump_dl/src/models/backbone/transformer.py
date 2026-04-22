from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..base import BaseBackbone
from ..layers import build_block
from ..layers.transformer import PositionEmbedding
from .registry import register_backbone
from ...utils.externals import ensure_torch

torch = ensure_torch()
nn = torch.nn


@register_backbone("transformer_sequence")
@register_backbone("transformer")
class TransformerSequenceBackbone(BaseBackbone):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_hidden_size: int | None = None,
        ffn_activation: str = "swiglu",
        norm_type: str = "rmsnorm",
        norm_eps: float = 1e-6,
        residual_scale: float = 1.0,
        qk_norm: bool = True,
        attention_gate: bool = True,
        max_seq_len: int = 4096,
        position_encoding: str = "rope",
        causal: bool = False,
        blocks: Sequence[Mapping[str, object]] | None = None,
    ) -> None:
        super().__init__()
        self.position_encoding = str(position_encoding).strip().lower()
        self.input_position = (
            PositionEmbedding(model_dim=hidden_size, max_seq_len=max_seq_len, mode="absolute")
            if self.position_encoding == "absolute" else None
        )
        if blocks is None:
            blocks = [
                {
                    "name": "transformer",
                    "num_heads": num_heads,
                    "dropout": dropout,
                    "ffn_hidden_size": ffn_hidden_size,
                    "ffn_activation": ffn_activation,
                    "norm_type": norm_type,
                    "norm_eps": norm_eps,
                    "residual_scale": residual_scale,
                    "qk_norm": qk_norm,
                    "attention_gate": attention_gate,
                    "max_seq_len": max_seq_len,
                    "position_encoding": "none" if self.position_encoding == "absolute" else self.position_encoding,
                    "causal": causal,
                }
                for _ in range(int(num_layers))
            ]
        self.blocks = nn.ModuleList(
            [build_block(block_cfg, hidden_size=int(hidden_size)) for block_cfg in blocks]
        )
        self.output_dim = int(hidden_size)

    def forward(self, x, *, padding_mask=None):
        if self.input_position is not None:
            x = self.input_position.add_to_input(x)
        for block in self.blocks:
            x = block(x, padding_mask=padding_mask)
        return x

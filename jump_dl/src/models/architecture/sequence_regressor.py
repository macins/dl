from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..base import BaseModel
from ..backbone import build_backbone
from ..encoder import build_encoder
from ..head import build_head
from ..registry import register_model


def _merge_config(base: dict[str, object], override: Mapping[str, object] | None) -> dict[str, object]:
    out = dict(base)
    if override:
        out.update(dict(override))
    return out


class _BaseSequenceRegressor(BaseModel):
    def __init__(
        self,
        *,
        numeric_feature_groups: Sequence[str] = ("continuous",),
        categorical_group_name: str = "category",
        categorical_cols: Sequence[str] = (),
        vocab_sizes: Mapping[str, int] | None = None,
        categorical_embedding_dim: int = 8,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        target_key: str = "ret_30min",
        encoder: Mapping[str, object] | None = None,
        backbone: Mapping[str, object] | None = None,
        head: Mapping[str, object] | None = None,
        default_backbone_name: str = "gru_sequence",
        default_backbone_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        encoder_cfg = _merge_config(
            {
                "name": "tabular_sequence",
                "numeric_feature_groups": numeric_feature_groups,
                "categorical_group_name": categorical_group_name,
                "categorical_cols": categorical_cols,
                "vocab_sizes": vocab_sizes,
                "categorical_embedding_dim": categorical_embedding_dim,
                "model_dim": hidden_size,
            },
            encoder,
        )
        self.encoder = build_encoder(encoder_cfg)

        backbone_cfg = _merge_config(
            {
                "name": default_backbone_name,
                "hidden_size": self.encoder.output_dim,
                "num_layers": num_layers,
                "dropout": dropout,
            },
            _merge_config(dict(default_backbone_kwargs or {}), backbone),
        )
        self.backbone = build_backbone(backbone_cfg)

        head_cfg = _merge_config(
            {
                "name": "sequence_regression",
                "input_dim": self.backbone.output_dim,
                "target_key": target_key,
                "output_dim": 1,
            },
            head,
        )
        self.head = build_head(head_cfg)

    def forward(self, batch: dict) -> dict:
        x = self.encoder(batch)
        x = self.backbone(x, padding_mask=batch.get("padding_mask"))
        return self.head(x)


@register_model("gru_sequence_regressor")
@register_model("modular_sequence_regressor")
class ModularSequenceRegressor(_BaseSequenceRegressor):
    pass


@register_model("transformer_sequence_regressor")
class TransformerSequenceRegressor(_BaseSequenceRegressor):
    def __init__(
        self,
        *,
        num_heads: int = 8,
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
        **kwargs,
    ) -> None:
        super().__init__(
            default_backbone_name="transformer_sequence",
            default_backbone_kwargs={
                "num_heads": num_heads,
                "ffn_hidden_size": ffn_hidden_size,
                "ffn_activation": ffn_activation,
                "norm_type": norm_type,
                "norm_eps": norm_eps,
                "residual_scale": residual_scale,
                "qk_norm": qk_norm,
                "attention_gate": attention_gate,
                "max_seq_len": max_seq_len,
                "position_encoding": position_encoding,
                "causal": causal,
            },
            **kwargs,
        )


GRUSequenceRegressor = ModularSequenceRegressor

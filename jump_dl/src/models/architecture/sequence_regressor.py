from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..base import BaseModel
from ..backbone import build_backbone
from ..encoder import build_encoder
from ..head import build_head
from ..registry import register_model
from ..layers import SymbolQueryDecoder, LongTermMemoryRead, PersistentMemoryBank, PrecomputedMemoryEncoder
from ..head.multi_horizon import MultiHorizonHeads, HorizonQueryDecoder


def _merge_config(
    base: dict[str, object],
    override: Mapping[str, object] | None,
) -> dict[str, object]:
    out = dict(base)
    if override:
        out.update(dict(override))
    return out


class _BaseSequenceRegressor(BaseModel):
    """
    Shape convention:

    Old symbol-day mode:
        encoder input:  batch["features"][group] = (B, T, F)
        encoder output: x = (B, T, D)
        mask:           padding_mask = (B, T)
        head output:    pred = (B, T, Y)

    New market-day / panel mode:
        encoder input:  batch["features"][group] = (B, N, T, F)
        encoder output: x = (B, N, T, D)
        mask:           padding_mask = (B, N, T)
        head output:    pred = (B, N, T, Y)

    This wrapper is intentionally layout-agnostic. The encoder/backbone/head
    should operate on the last feature dimension and preserve leading dims.
    """

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
        symbol_query_decoder: Mapping[str, object] | None = None,
        multi_horizon: Mapping[str, object] | None = None,
        long_term_memory: Mapping[str, object] | None = None,
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

        sqd_cfg = dict(symbol_query_decoder or {})
        self.symbol_query_decoder = None
        if bool(sqd_cfg.get("enabled", False)):
            sqd_cfg.pop("enabled", None)
            sqd_cfg.pop("topk_indices_path", None)
            sqd_cfg.setdefault("d_model", self.backbone.output_dim)
            self.symbol_query_decoder = SymbolQueryDecoder(**sqd_cfg)

        ltm_cfg = dict(long_term_memory or {})
        self.long_term_memory_cfg = ltm_cfg
        self.long_term_memory_enabled = bool(ltm_cfg.get("enabled", False))
        self.long_term_memory_read = None
        self.long_term_memory_batch_key = str(ltm_cfg.get("precomputed", {}).get("batch_key", "rolling_memory"))
        self.long_term_memory_mode = str(ltm_cfg.get("mode", "persistent"))
        if self.long_term_memory_enabled:
            d_model = int(ltm_cfg.get("d_model") or self.backbone.output_dim)
            memory_levels = list(ltm_cfg.get("memory_levels", ["market"]))
            persistent_bank = None
            if self.long_term_memory_mode in {"persistent", "persistent_plus_precomputed"}:
                persistent_bank = PersistentMemoryBank(
                    d_model=d_model,
                    num_market_slots=int(ltm_cfg.get("num_market_slots", 16)),
                    num_symbol_slots=int(ltm_cfg.get("num_symbol_slots", 0)),
                    num_symbols=ltm_cfg.get("num_symbols"),
                    memory_levels=memory_levels,
                )
            precomputed_encoder = None
            if self.long_term_memory_mode in {"precomputed_context", "persistent_plus_precomputed"}:
                pre_cfg = dict(ltm_cfg.get("precomputed", {}))
                sdim = pre_cfg.get("summary_dim")
                if sdim is not None:
                    precomputed_encoder = PrecomputedMemoryEncoder(
                        summary_dim=int(sdim),
                        d_model=d_model,
                        num_summary_slots=int(pre_cfg.get("num_summary_slots", 4)),
                        encoder_type=str(pre_cfg.get("encoder_type", "mlp")),
                        pooling=str(pre_cfg.get("pooling", "mean")),
                        include_market_summary=bool(pre_cfg.get("include_market_summary", True)),
                        include_symbol_summary=bool(pre_cfg.get("include_symbol_summary", True)),
                    )
            self.long_term_memory_read = LongTermMemoryRead(
                d_model=d_model,
                num_heads=int(ltm_cfg.get("num_heads", 4)),
                dropout=float(ltm_cfg.get("dropout", 0.0)),
                residual_init=float(ltm_cfg.get("residual_init", 0.0)),
                use_layer_norm=bool(ltm_cfg.get("use_layer_norm", True)),
                use_gate=bool(ltm_cfg.get("use_gate", True)),
                gate_type=str(ltm_cfg.get("gate_type", "scalar")),
                read_mode=str(ltm_cfg.get("read_mode", "gated_cross_attn")),
                persistent_bank=persistent_bank,
                precomputed_encoder=precomputed_encoder,
            )

        mh_cfg = dict(multi_horizon or {})
        self.multi_horizon_enabled = bool(mh_cfg.get("enabled", False))
        self.multi_horizon_horizons = [int(h) for h in mh_cfg.get("horizons", [30])]
        self.multi_horizon_main = int(mh_cfg.get("main_horizon", 30))
        self.multi_horizon_mode = str(mh_cfg.get("mode", "aux_heads"))
        self.mh_decoder = None
        self.mh_heads = None
        if self.multi_horizon_enabled:
            if self.multi_horizon_mode == "horizon_query_decoder":
                dcfg = dict(mh_cfg.get("horizon_query_decoder", {}))
                self.mh_decoder = HorizonQueryDecoder(
                    d_model=self.backbone.output_dim,
                    horizons=self.multi_horizon_horizons,
                    num_heads=int(dcfg.get("num_heads", 4)),
                    dropout=float(dcfg.get("dropout", 0.0)),
                    use_horizon_embedding=bool(dcfg.get("use_horizon_embedding", True)),
                    use_layer_norm=bool(dcfg.get("use_layer_norm", True)),
                    residual_init=float(dcfg.get("residual_init", 0.0)),
                    attend_mode=str(dcfg.get("attend_mode", "own_history")),
                )
            else:
                self.mh_heads = MultiHorizonHeads(
                    d_model=self.backbone.output_dim,
                    horizons=self.multi_horizon_horizons,
                    output_dim=1,
                )

    def forward(self, batch: dict) -> dict:
        padding_mask = batch.get("padding_mask")

        x = self.encoder(batch)
        x = self.backbone(x, padding_mask=padding_mask)
        if self.symbol_query_decoder is not None:
            x = self.symbol_query_decoder(x, symbol_ids=batch.get("symbol_ids"))
        if self.long_term_memory_enabled and self.long_term_memory_read is not None:
            precomputed_memory = None
            if self.long_term_memory_mode in {"precomputed_context", "persistent_plus_precomputed"}:
                if self.long_term_memory_batch_key not in batch:
                    raise KeyError(f"Missing required precomputed memory batch key: {self.long_term_memory_batch_key}")
                precomputed_memory = batch[self.long_term_memory_batch_key]
            x = self.long_term_memory_read(x, symbol_ids=batch.get("symbol_ids"), precomputed_memory=precomputed_memory)
        out = self.head(x)

        if self.multi_horizon_enabled:
            if self.mh_decoder is not None:
                pred_by_hz, mh_aux = self.mh_decoder(x)
            else:
                pred_by_hz = self.mh_heads(x)
                mh_aux = {}
            if self.multi_horizon_main not in pred_by_hz:
                raise ValueError(f"main_horizon={self.multi_horizon_main} not in horizons={self.multi_horizon_horizons}")
            out["preds"][self.head.target_key] = pred_by_hz[self.multi_horizon_main]
            out["pred_by_horizon"] = pred_by_hz
            out.setdefault("aux_metrics", {}).update({"multi_horizon_enabled": 1.0})
            if mh_aux:
                out.setdefault("aux", {}).update(mh_aux)

        aux_losses = getattr(self.backbone, "last_aux_losses", {})
        aux_metrics = dict(getattr(self.backbone, "last_aux_metrics", {}))
        if self.symbol_query_decoder is not None:
            aux_metrics.update(self.symbol_query_decoder.get_aux_stats())
        if self.long_term_memory_enabled and self.long_term_memory_read is not None:
            aux_metrics.update(self.long_term_memory_read.get_aux_stats())

        if aux_losses:
            out["aux_losses"] = aux_losses
        if aux_metrics:
            out["aux_metrics"] = aux_metrics

        return out


@register_model("gru_sequence_regressor")
@register_model("modular_sequence_regressor")
class ModularSequenceRegressor(_BaseSequenceRegressor):
    pass


@register_model("transformer_sequence_regressor")
@register_model("transformer_panel_regressor")
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

        # MoE defaults
        use_moe: bool = False,
        num_experts: int = 4,
        expert_hidden_size: int | None = None,
        top_k: int = 2,
        aux_loss_weight: float = 1e-2,
        router_z_loss_weight: float = 1e-3,
        shared_experts: int = 0,

        # new: custom block internal layout
        sublayers: Sequence[str | Mapping[str, Any]] | None = None,

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
                "use_moe": use_moe,
                "num_experts": num_experts,
                "expert_hidden_size": expert_hidden_size,
                "top_k": top_k,
                "aux_loss_weight": aux_loss_weight,
                "router_z_loss_weight": router_z_loss_weight,
                "shared_experts": shared_experts,
                "sublayers": sublayers,
            },
            **kwargs,
        )


GRUSequenceRegressor = ModularSequenceRegressor
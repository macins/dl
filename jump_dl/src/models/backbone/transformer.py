from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

from ..base import BaseBackbone
from ..layers import build_block
from ..layers.transformer import PositionEmbedding
from .registry import register_backbone
from ...utils.externals import ensure_torch

torch = ensure_torch()
nn = torch.nn


def _normalize_cross_symbol_layer_type(layer_type: str) -> str:
    """
    Normalize top-level convenience cross-symbol layer type.

    Supported canonical types:
        - cross_symbol_attention
        - cross_symbol_film
        - cross_symbol_memory_attention
        - cross_symbol_memory_film
    """
    layer_type = str(layer_type).strip().lower()

    aliases = {
        # Dense cross-symbol attention.
        "attention": "cross_symbol_attention",
        "attn": "cross_symbol_attention",
        "cross_attention": "cross_symbol_attention",
        "cross-attention": "cross_symbol_attention",
        "cross_symbol_attention": "cross_symbol_attention",
        "cross-symbol-attention": "cross_symbol_attention",
        "cross_symbol_attn": "cross_symbol_attention",
        "cross-symbol-attn": "cross_symbol_attention",
        "symbol_attention": "cross_symbol_attention",
        "symbol_attn": "cross_symbol_attention",

        # Lagged cross-symbol attention.
        "cross_symbol_lagged_attention": "cross_symbol_lagged_attention",
        "cross-symbol-lagged-attention": "cross_symbol_lagged_attention",
        "cross_lagged_attention": "cross_symbol_lagged_attention",
        "cross-lagged-attention": "cross_symbol_lagged_attention",
        "lagged_cross_attention": "cross_symbol_lagged_attention",
        "lagged_cross_symbol_attention": "cross_symbol_lagged_attention",
        "lead_lag_attention": "cross_symbol_lagged_attention",
        "lead-lag-attention": "cross_symbol_lagged_attention",
        
        # Dense cross-symbol FiLM.
        "film": "cross_symbol_film",
        "cross_film": "cross_symbol_film",
        "cross-film": "cross_symbol_film",
        "symbol_film": "cross_symbol_film",
        "symbol-film": "cross_symbol_film",
        "cross_symbol_film": "cross_symbol_film",
        "cross-symbol-film": "cross_symbol_film",

        # Memory cross-symbol attention.
        "memory": "cross_symbol_memory_attention",
        "memory_attention": "cross_symbol_memory_attention",
        "memory-attention": "cross_symbol_memory_attention",
        "cross_memory": "cross_symbol_memory_attention",
        "cross-memory": "cross_symbol_memory_attention",
        "cross_memory_attention": "cross_symbol_memory_attention",
        "cross-memory-attention": "cross_symbol_memory_attention",
        "symbol_memory_attention": "cross_symbol_memory_attention",
        "symbol-memory-attention": "cross_symbol_memory_attention",
        "memory_cross_attention": "cross_symbol_memory_attention",
        "memory-cross-attention": "cross_symbol_memory_attention",
        "memory_cross_symbol_attention": "cross_symbol_memory_attention",
        "memory-cross-symbol-attention": "cross_symbol_memory_attention",
        "cross_symbol_memory_attention": "cross_symbol_memory_attention",
        "cross-symbol-memory-attention": "cross_symbol_memory_attention",

        # Memory cross-symbol FiLM.
        "memory_film": "cross_symbol_memory_film",
        "memory-film": "cross_symbol_memory_film",
        "cross_memory_film": "cross_symbol_memory_film",
        "cross-memory-film": "cross_symbol_memory_film",
        "symbol_memory_film": "cross_symbol_memory_film",
        "symbol-memory-film": "cross_symbol_memory_film",
        "memory_cross_film": "cross_symbol_memory_film",
        "memory-cross-film": "cross_symbol_memory_film",
        "memory_cross_symbol_film": "cross_symbol_memory_film",
        "memory-cross-symbol-film": "cross_symbol_memory_film",
        "cross_symbol_memory_film": "cross_symbol_memory_film",
        "cross-symbol-memory-film": "cross_symbol_memory_film",
    }

    normalized = aliases.get(layer_type, layer_type)

    if normalized not in {
        "cross_symbol_attention",
        "cross_symbol_lagged_attention",
        "cross_symbol_film",
        "cross_symbol_memory_attention",
        "cross_symbol_memory_film",
    }:
        raise ValueError(
            f"Unknown cross_symbol_layer_type={layer_type!r}. "
            "Supported: attention, film, memory_attention, memory_film."
        )

    return normalized


def _make_default_sublayers(
    *,
    use_moe: bool,
    use_cross_symbol: bool,
    cross_symbol_layer_type: str,
    cross_symbol_kwargs: Mapping[str, object] | None,
) -> list[dict[str, Any]] | None:
    """
    Build a default sublayer layout only when the user asks for a top-level
    cross-symbol layer.

    Default old behavior is preserved:
        sublayers=None and use_cross_symbol=False
            -> let TransformerEncoderBlock use its own default:
               attention -> ffn/moe_ffn

    Convenience behavior:
        use_cross_symbol=True, cross_symbol_layer_type="attention"
            -> attention -> cross_symbol_attention -> ffn/moe_ffn

        use_cross_symbol=True, cross_symbol_layer_type="film"
            -> attention -> cross_symbol_film -> ffn/moe_ffn

        use_cross_symbol=True, cross_symbol_layer_type="memory_attention"
            -> attention -> cross_symbol_memory_attention -> ffn/moe_ffn

        use_cross_symbol=True, cross_symbol_layer_type="memory_film"
            -> attention -> cross_symbol_memory_film -> ffn/moe_ffn
    """
    if not use_cross_symbol:
        return None

    cross_type = _normalize_cross_symbol_layer_type(cross_symbol_layer_type)

    cross_spec: dict[str, Any] = {"type": cross_type}
    if cross_symbol_kwargs is not None:
        cross_spec.update(dict(cross_symbol_kwargs))

    return [
        {"type": "attention"},
        cross_spec,
        {"type": "moe_ffn" if use_moe else "ffn"},
    ]


@register_backbone("transformer_sequence")
@register_backbone("transformer")
class TransformerSequenceBackbone(BaseBackbone):
    """
    Shape convention:

    x:
        (B, T, D)
        or
        (B, N, T, D)

    padding_mask:
        (B, T)
        or
        (B, N, T)

    This backbone itself is mostly layout-agnostic. Each transformer block is
    responsible for attending over the last sequence dimension T.

    Explicit sublayer interface:

        sublayers:
          - type: attention
            causal: true
          - type: cross_symbol_memory_film
            mask_self: true
            memory_type: gru
            memory_hidden_size: 256
            film_mode: residual_norm
          - type: ffn

    Top-level convenience interface:

        use_cross_symbol: true
        cross_symbol_layer_type: memory_film
        cross_symbol_kwargs:
          mask_self: true
          memory_type: gru
          memory_hidden_size: 256
          memory_value_source: memory
          film_mode: residual_norm
          use_beta: true
          zero_init: true

    If both explicit sublayers and use_cross_symbol are provided, explicit
    sublayers win.

    Attention projection sharing:

        attention_projection_sharing: none
        attention_projection_sharing: v
        attention_projection_sharing: kv
        attention_projection_sharing: qkv

    Sharing is block-local:
        block_0 has its own shared projection set;
        block_1 has a different shared projection set;
        etc.
    """

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
        residual_scale: Optional[float] = 1,
        qk_norm: bool = True,
        attention_gate: bool = True,
        max_seq_len: int = 4096,
        position_encoding: str = "rope",
        causal: bool = False,

        use_moe: bool = False,
        num_experts: int = 4,
        expert_hidden_size: int | None = None,
        top_k: int = 2,
        aux_loss_weight: float = 1e-2,
        router_z_loss_weight: float = 1e-3,
        shared_experts: int = 0,

        # Explicit sublayer interface.
        # If provided, this is passed to every default transformer block.
        sublayers: Sequence[str | Mapping[str, object]] | None = None,

        # Top-level convenience interface for inserting one cross-symbol sublayer
        # into every default block:
        #
        #   attention -> cross-symbol layer -> ffn/moe_ffn
        #
        # Supported cross_symbol_layer_type:
        #   attention
        #   film
        #   memory_attention
        #   memory_film
        #
        # Explicit `sublayers` takes priority over this.
        use_cross_symbol: bool = False,
        cross_symbol_layer_type: str = "attention",
        cross_symbol_kwargs: Mapping[str, object] | None = None,

        # Passed to every default TransformerEncoderBlock.
        # Supported by the transformer block:
        #   "none", "v", "kv", "qkv", etc.
        #
        # This is block-local sharing, not cross-layer sharing.
        attention_projection_sharing: str | Sequence[str] | None = "none",
        codebook_enabled: bool = False,
        codebook_num_codes: int = 128,
        codebook_num_heads: int = 4,
        codebook_dropout: float = 0.0,
        codebook_topk: int | None = None,
        codebook_temperature: float = 1.0,
        codebook_residual_gate_init: float = 0.0,
        codebook_use_layernorm: bool = True,
        codebook_share_kv: bool = False,
        codebook_position: str = "after_ffn",

        # Register-token interface.
        # Default keeps the old behavior unchanged.
        use_register_tokens: bool = False,
        num_register_tokens: int = 1,
        register_axis: str = "symbol",
        return_register_tokens: bool = False,

        # If provided, this fully overrides the default num_layers repeated blocks.
        #
        # If use_register_tokens=True, the register-token kwargs are injected into
        # each block config unless that block already explicitly sets them.
        #
        # If a custom block does not define `sublayers`, then global `sublayers`
        # or the top-level cross-symbol convenience layout may be used as defaults.
        #
        # If a custom block does not define `attention_projection_sharing`, then
        # the global value is used as default.
        blocks: Sequence[Mapping[str, object]] | None = None,
    ) -> None:
        super().__init__()

        self.last_aux_losses: dict[str, torch.Tensor] = {}
        self.last_aux_metrics: dict[str, float] = {}

        self.hidden_size = int(hidden_size)
        self.output_dim = int(hidden_size)

        self.position_encoding = str(position_encoding).strip().lower()
        self.attention_projection_sharing = attention_projection_sharing

        self.use_register_tokens = bool(use_register_tokens)
        self.num_register_tokens = int(num_register_tokens)
        self.register_axis = str(register_axis).strip().lower()
        self.return_register_tokens = bool(return_register_tokens)

        if self.use_register_tokens:
            if self.register_axis != "symbol":
                raise ValueError(
                    "Currently only register_axis='symbol' is supported. "
                    "Expected input shape (B,N,T,D)."
                )
            if self.num_register_tokens <= 0:
                raise ValueError(
                    f"num_register_tokens must be positive, got {self.num_register_tokens}"
                )

        # For absolute position encoding, add it once at backbone input.
        # For rope/alibi, each attention layer handles it internally.
        self.input_position = (
            PositionEmbedding(
                model_dim=self.hidden_size,
                max_seq_len=max_seq_len,
                mode="absolute",
            )
            if self.position_encoding == "absolute"
            else None
        )

        if residual_scale is None:
            # Standard depth-aware residual scaling.
            # If each block has multiple sublayers, this is still a reasonable
            # default, but you may want to set residual_scale explicitly.
            residual_scale = (2 * int(num_layers)) ** (-0.5)

        register_cfg: dict[str, Any] = {
            "use_register_tokens": self.use_register_tokens,
            "num_register_tokens": self.num_register_tokens,
            "register_axis": self.register_axis,
            "return_register_tokens": self.return_register_tokens,
        }

        default_cross_sublayers = _make_default_sublayers(
            use_moe=use_moe,
            use_cross_symbol=bool(use_cross_symbol),
            cross_symbol_layer_type=cross_symbol_layer_type,
            cross_symbol_kwargs=cross_symbol_kwargs,
        )

        # Explicit sublayers win over convenience use_cross_symbol.
        default_sublayers: Sequence[str | Mapping[str, object]] | None
        default_sublayers = sublayers if sublayers is not None else default_cross_sublayers

        if blocks is None:
            block_cfgs: list[dict[str, Any]] = []

            for _ in range(int(num_layers)):
                cfg: dict[str, Any] = {
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

                    # absolute is applied once at backbone input,
                    # so inner attention blocks should not apply absolute again.
                    "position_encoding": (
                        "none"
                        if self.position_encoding == "absolute"
                        else self.position_encoding
                    ),

                    "causal": causal,
                    "use_moe": use_moe,
                    "num_experts": num_experts,
                    "expert_hidden_size": expert_hidden_size,
                    "top_k": top_k,
                    "aux_loss_weight": aux_loss_weight,
                    "router_z_loss_weight": router_z_loss_weight,
                    "shared_experts": shared_experts,

                    # Forward block-local Q/K/V sharing mode to each block.
                    "attention_projection_sharing": attention_projection_sharing,
                    "codebook_enabled": codebook_enabled,
                    "codebook_num_codes": codebook_num_codes,
                    "codebook_num_heads": codebook_num_heads,
                    "codebook_dropout": codebook_dropout,
                    "codebook_topk": codebook_topk,
                    "codebook_temperature": codebook_temperature,
                    "codebook_residual_gate_init": codebook_residual_gate_init,
                    "codebook_use_layernorm": codebook_use_layernorm,
                    "codebook_share_kv": codebook_share_kv,
                    "codebook_position": codebook_position,

                    **register_cfg,
                }

                if default_sublayers is not None:
                    cfg["sublayers"] = default_sublayers

                block_cfgs.append(cfg)

        else:
            # Fully custom per-block configs.
            # Each block can define its own sublayers.
            #
            # Per-block explicit values win over global defaults.
            block_cfgs = []

            for block_cfg in blocks:
                cfg = dict(block_cfg)

                cfg.setdefault("name", "transformer")

                # Global register-token args are defaults only.
                # Per-block explicit values win.
                for key, value in register_cfg.items():
                    cfg.setdefault(key, value)

                # If the custom block did not specify sublayers, allow the
                # global explicit sublayers or top-level cross-symbol layout
                # to act as a default.
                if default_sublayers is not None:
                    cfg.setdefault("sublayers", default_sublayers)

                # Global attention projection sharing is a default only.
                # Per-block explicit value wins.
                cfg.setdefault(
                    "attention_projection_sharing",
                    attention_projection_sharing,
                )
                cfg.setdefault("codebook_enabled", codebook_enabled)
                cfg.setdefault("codebook_num_codes", codebook_num_codes)
                cfg.setdefault("codebook_num_heads", codebook_num_heads)
                cfg.setdefault("codebook_dropout", codebook_dropout)
                cfg.setdefault("codebook_topk", codebook_topk)
                cfg.setdefault("codebook_temperature", codebook_temperature)
                cfg.setdefault("codebook_residual_gate_init", codebook_residual_gate_init)
                cfg.setdefault("codebook_use_layernorm", codebook_use_layernorm)
                cfg.setdefault("codebook_share_kv", codebook_share_kv)
                cfg.setdefault("codebook_position", codebook_position)

                block_cfgs.append(cfg)

        self.blocks = nn.ModuleList(
            [
                build_block(
                    block_cfg,
                    hidden_size=self.hidden_size,
                )
                for block_cfg in block_cfgs
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x:
            (B, T, D)
            or
            (B, N, T, D)

        padding_mask:
            (B, T)
            or
            (B, N, T)
        """
        if x.ndim not in (3, 4):
            raise ValueError(
                f"TransformerSequenceBackbone expects x with shape "
                f"(B,T,D) or (B,N,T,D), got {tuple(x.shape)}."
            )

        if self.use_register_tokens and x.ndim != 4:
            raise ValueError(
                "use_register_tokens=True requires x with shape (B,N,T,D), "
                f"got {tuple(x.shape)}."
            )

        if padding_mask is not None and padding_mask.shape != x.shape[:-1]:
            raise ValueError(
                f"padding_mask shape mismatch: expected {tuple(x.shape[:-1])}, "
                f"got {tuple(padding_mask.shape)}."
            )

        self.last_aux_losses = {}
        self.last_aux_metrics = {}

        if self.input_position is not None:
            x = self.input_position.add_to_input(x)

        for block_idx, block in enumerate(self.blocks):
            x = block(x, padding_mask=padding_mask)

            for name, value in getattr(block, "last_aux_losses", {}).items():
                # Aggregate same-name aux losses across blocks.
                self.last_aux_losses[name] = self.last_aux_losses.get(name, 0) + value

                # Also keep block-specific version.
                self.last_aux_losses[f"block_{block_idx}_{name}"] = value

            for name, value in getattr(block, "last_aux_metrics", {}).items():
                self.last_aux_metrics[f"block_{block_idx}_{name}"] = value

        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1).to(dtype=x.dtype)

        return x

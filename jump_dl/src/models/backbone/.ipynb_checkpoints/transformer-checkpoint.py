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

    This backbone itself is layout-agnostic. Each transformer block is responsible
    for attending over the last sequence dimension T.
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

        # Passed to every default transformer block.
        sublayers: Sequence[str | Mapping[str, object]] | None = None,

        # Register-token interface.
        # Default keeps the old behavior unchanged.
        use_register_tokens: bool = False,
        num_register_tokens: int = 1,
        register_axis: str = "symbol",
        return_register_tokens: bool = False,

        # If provided, this fully overrides the default num_layers repeated blocks.
        # If use_register_tokens=True, the register-token kwargs are injected into
        # each block config unless that block already explicitly sets them.
        blocks: Sequence[Mapping[str, object]] | None = None,
    ) -> None:
        super().__init__()

        self.last_aux_losses: dict[str, torch.Tensor] = {}
        self.last_aux_metrics: dict[str, float] = {}

        self.hidden_size = int(hidden_size)
        self.output_dim = int(hidden_size)

        self.position_encoding = str(position_encoding).strip().lower()

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
            # If each block now has multiple sublayers, this is still a reasonable default,
            # but you may want to set residual_scale explicitly.
            residual_scale = (2 * int(num_layers)) ** (-0.5)

        register_cfg: dict[str, Any] = {
            "use_register_tokens": self.use_register_tokens,
            "num_register_tokens": self.num_register_tokens,
            "register_axis": self.register_axis,
            "return_register_tokens": self.return_register_tokens,
        }

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

                    **register_cfg,
                }

                if sublayers is not None:
                    cfg["sublayers"] = sublayers

                block_cfgs.append(cfg)

        else:
            # Fully custom per-block configs.
            # Each block can define its own sublayers.
            block_cfgs = []

            for block_cfg in blocks:
                cfg = dict(block_cfg)

                # Global register-token args are defaults only.
                # Per-block explicit values win.
                for key, value in register_cfg.items():
                    cfg.setdefault(key, value)

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
                # aggregate same-name aux losses across blocks
                self.last_aux_losses[name] = self.last_aux_losses.get(name, 0) + value

                # also keep block-specific version
                self.last_aux_losses[f"block_{block_idx}_{name}"] = value

            for name, value in getattr(block, "last_aux_metrics", {}).items():
                self.last_aux_metrics[f"block_{block_idx}_{name}"] = value

        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1).to(dtype=x.dtype)

        return x
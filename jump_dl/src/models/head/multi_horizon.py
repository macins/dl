from __future__ import annotations

from collections.abc import Sequence

from ...utils.externals import ensure_torch

torch = ensure_torch()
nn = torch.nn


class MultiHorizonHeads(nn.Module):
    """Simple per-horizon linear readout heads for [B, N, T, D] hidden states."""

    def __init__(self, d_model: int, horizons: Sequence[int], output_dim: int = 1) -> None:
        super().__init__()
        self.horizons = [int(h) for h in horizons]
        self.heads = nn.ModuleDict(
            {str(h): nn.Linear(int(d_model), int(output_dim), bias=False) for h in self.horizons}
        )

    def forward(self, h: torch.Tensor) -> dict[int, torch.Tensor]:
        return {hz: self.heads[str(hz)](h) for hz in self.horizons}


class HorizonQueryDecoder(nn.Module):
    """Horizon-conditioned causal temporal decoder (own-history only)."""

    def __init__(
        self,
        d_model: int,
        horizons: Sequence[int],
        num_heads: int,
        dropout: float = 0.0,
        use_horizon_embedding: bool = True,
        use_layer_norm: bool = True,
        residual_init: float = 0.0,
        attend_mode: str = "own_history",
    ) -> None:
        super().__init__()
        if attend_mode != "own_history":
            raise ValueError(f"Unsupported attend_mode={attend_mode!r}. Only 'own_history' is implemented.")
        self.horizons = [int(h) for h in horizons]
        self.use_horizon_embedding = bool(use_horizon_embedding)
        self.norm = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.residual_alpha = nn.Parameter(torch.tensor(float(residual_init)))
        if self.use_horizon_embedding:
            self.horizon_embed = nn.Embedding(len(self.horizons), d_model)
        else:
            self.horizon_embed = None
        self.heads = MultiHorizonHeads(d_model=d_model, horizons=self.horizons, output_dim=1)

    def forward(self, h: torch.Tensor) -> tuple[dict[int, torch.Tensor], dict[str, torch.Tensor]]:
        b, n, t, d = h.shape
        x = h.reshape(b * n, t, d)
        k = self.k_proj(self.norm(x))
        v = self.v_proj(self.norm(x))
        causal_mask = torch.triu(torch.ones(t, t, device=h.device, dtype=torch.bool), diagonal=1)
        pred_by_hz: dict[int, torch.Tensor] = {}
        states: dict[str, torch.Tensor] = {}
        for idx, hz in enumerate(self.horizons):
            q = self.q_proj(self.norm(x))
            if self.horizon_embed is not None:
                q = q + self.horizon_embed.weight[idx].view(1, 1, -1)
            attn_out, _ = self.attn(q, k, v, attn_mask=causal_mask, need_weights=False)
            z = x + self.residual_alpha * attn_out
            z4 = z.reshape(b, n, t, d)
            pred_by_hz[hz] = self.heads.heads[str(hz)](z4)
            states[f"horizon_state_{hz}"] = z4
        return pred_by_hz, states

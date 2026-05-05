from __future__ import annotations

import math

from ..base import BaseHead
from .registry import register_head
from ...utils.externals import ensure_torch

torch = ensure_torch()
nn = torch.nn
F = torch.nn.functional


def _inverse_softplus(x: float) -> float:
    x = max(float(x), 1e-6)
    return math.log(math.expm1(x))


@register_head("conditional_latent_factor_mog")
class ConditionalLatentFactorMoGHead(BaseHead):
    """Conditional latent factor MoG head.

    Sigma init guidance:
    - For normalized/z-scored targets, `init_factor_sigma` and `init_residual_sigma` around 1.0 are reasonable.
    - For raw decimal returns, use smaller target-scale values (e.g. ~1e-3).
    """

    def __init__(
        self,
        *,
        input_dim: int,
        target_key: str = "ret_30min",
        output_dim: int = 1,
        num_factors: int = 16,
        num_components: int = 4,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        final_step_only: bool = False,
        three_dim_layout: str = "BTD",
        exposure_normalize: bool = True,
        exposure_norm_eps: float = 1e-6,
        use_layernorm: bool = True,
        min_factor_sigma: float = 1e-4,
        min_residual_sigma: float = 1e-4,
        max_sigma: float | None = None,
        market_pooling: str = "mean",
        detach_market_pool: bool = False,
        init_factor_sigma: float = 1.0,
        init_residual_sigma: float = 1.0,
        factor_mu_init_std: float = 1e-3,
        **kwargs,
    ) -> None:
        super().__init__()
        _ = kwargs
        self.input_dim = int(input_dim)
        self.target_key = str(target_key)
        self.output_dim = int(output_dim)
        if self.output_dim != 1:
            raise ValueError(f"ConditionalLatentFactorMoGHead supports output_dim=1 only, got {self.output_dim}.")
        self.num_factors = int(num_factors)
        self.num_components = int(num_components)
        self.final_step_only = bool(final_step_only)
        self.three_dim_layout = str(three_dim_layout).upper()
        if self.three_dim_layout not in {"BTD", "BND"}:
            raise ValueError(f"three_dim_layout must be 'BTD' or 'BND', got {three_dim_layout!r}.")
        self.exposure_normalize = bool(exposure_normalize)
        self.exposure_norm_eps = float(exposure_norm_eps)
        self.min_factor_sigma = float(min_factor_sigma)
        self.min_residual_sigma = float(min_residual_sigma)
        self.max_sigma = None if max_sigma is None else float(max_sigma)
        self.market_pooling = str(market_pooling)
        self.detach_market_pool = bool(detach_market_pool)
        inner = int(hidden_dim or input_dim)

        self.norm = nn.LayerNorm(self.input_dim) if use_layernorm else nn.Identity()
        self.exposure_mlp = nn.Sequential(nn.Linear(self.input_dim, inner), nn.GELU(), nn.Dropout(dropout), nn.Linear(inner, self.num_factors))
        self.residual_mlp = nn.Sequential(nn.Linear(self.input_dim, inner), nn.GELU(), nn.Dropout(dropout), nn.Linear(inner, 1))

        if self.market_pooling not in {"mean", "attention"}:
            raise ValueError(f"Unsupported market_pooling={self.market_pooling!r}")
        if self.market_pooling == "attention":
            self.attn_pool = nn.Linear(self.input_dim, inner)
            self.attn_vec = nn.Linear(inner, 1, bias=False)
        else:
            self.attn_pool = None
            self.attn_vec = None

        self.mix_proj = nn.Linear(self.input_dim, self.num_components)
        self.factor_mu_proj = nn.Linear(self.input_dim, self.num_components * self.num_factors)
        self.factor_sigma_proj = nn.Linear(self.input_dim, self.num_components * self.num_factors)

        nn.init.normal_(self.factor_mu_proj.weight, mean=0.0, std=float(factor_mu_init_std))
        nn.init.zeros_(self.factor_mu_proj.bias)
        nn.init.constant_(self.factor_sigma_proj.bias, _inverse_softplus(max(init_factor_sigma - self.min_factor_sigma, 1e-6)))
        nn.init.constant_(self.residual_mlp[-1].bias, _inverse_softplus(max(init_residual_sigma - self.min_residual_sigma, 1e-6)))

    def _to_bntd(self, x: torch.Tensor) -> tuple[torch.Tensor, bool, bool]:
        if x.ndim == 4:
            return x, False, False
        if x.ndim == 3:
            if self.three_dim_layout == "BTD":
                return x.unsqueeze(1), True, False
            if self.three_dim_layout == "BND":
                return x.unsqueeze(2), False, True
        if x.ndim == 2:
            return x.unsqueeze(1).unsqueeze(2), True, True
        raise ValueError(
            "ConditionalLatentFactorMoGHead expects input with shape [B,N,T,D], [B,T,D] (three_dim_layout='BTD'), "
            "[B,N,D] (three_dim_layout='BND'), or [B,D]. "
            f"Got shape={tuple(x.shape)} with three_dim_layout={self.three_dim_layout!r}."
        )

    def _market_state(self, h: torch.Tensor) -> torch.Tensor:
        # TODO: support mask-aware market pooling if batches contain padded/invalid symbols.
        if self.market_pooling == "mean":
            c = h.mean(dim=1)
        else:
            s = self.attn_vec(torch.tanh(self.attn_pool(h))).squeeze(-1)
            a = torch.softmax(s, dim=1)
            c = torch.einsum("bnt,bntd->btd", a, h)
        return c.detach() if self.detach_market_pool else c

    def forward(self, x: torch.Tensor) -> dict:
        h, squeezed_n, squeezed_t = self._to_bntd(x)
        h = self.norm(h)
        if self.final_step_only:
            h = h[:, :, -1:, :]

        b, n, t, _ = h.shape
        exposure = self.exposure_mlp(h)
        if self.exposure_normalize and n > 1:
            mean = exposure.mean(dim=1, keepdim=True)
            var = (exposure - mean).pow(2).mean(dim=1, keepdim=True)
            exposure = (exposure - mean) / torch.sqrt(var + self.exposure_norm_eps)

        residual_sigma = F.softplus(self.residual_mlp(h)).squeeze(-1) + self.min_residual_sigma
        c = self._market_state(h)
        mix_logits = self.mix_proj(c)
        mix_probs = torch.softmax(mix_logits, dim=-1)
        factor_mu = self.factor_mu_proj(c).view(b, t, self.num_components, self.num_factors)
        factor_sigma = F.softplus(self.factor_sigma_proj(c)).view(b, t, self.num_components, self.num_factors) + self.min_factor_sigma

        if self.max_sigma is not None:
            residual_sigma = residual_sigma.clamp(max=self.max_sigma)
            factor_sigma = factor_sigma.clamp(max=self.max_sigma)

        component_pred = torch.einsum("bntp,btkp->bntk", exposure, factor_mu)
        pred = torch.einsum("bntk,btk->bnt", component_pred, mix_probs)

        canonical_pred = pred
        canonical_component_pred = component_pred
        canonical_exposure = exposure
        canonical_residual_sigma = residual_sigma
        canonical_factor_mu = factor_mu
        canonical_factor_sigma = factor_sigma
        canonical_mix_logits = mix_logits
        canonical_mix_probs = mix_probs

        if self.final_step_only:
            canonical_pred = canonical_pred.squeeze(2)
            canonical_component_pred = canonical_component_pred.squeeze(2)
            canonical_exposure = canonical_exposure.squeeze(2)
            canonical_residual_sigma = canonical_residual_sigma.squeeze(2)
            canonical_factor_mu = canonical_factor_mu.squeeze(1)
            canonical_factor_sigma = canonical_factor_sigma.squeeze(1)
            canonical_mix_logits = canonical_mix_logits.squeeze(1)
            canonical_mix_probs = canonical_mix_probs.squeeze(1)

        public_pred = canonical_pred
        if squeezed_n and public_pred.ndim >= 2:
            public_pred = public_pred.squeeze(1)
        if squeezed_t and public_pred.ndim >= 2:
            public_pred = public_pred.squeeze(-1)

        fm = {
            "pred": canonical_pred,
            "component_pred": canonical_component_pred,
            "exposure": canonical_exposure,
            "factor_mu": canonical_factor_mu,
            "factor_sigma": canonical_factor_sigma,
            "mix_logits": canonical_mix_logits,
            "mix_probs": canonical_mix_probs,
            "residual_sigma": canonical_residual_sigma,
        }
        return {"preds": {self.target_key: public_pred}, "pred": public_pred, "factor_mog": fm}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_factors={self.num_factors}, num_components={self.num_components}, "
            f"market_pooling={self.market_pooling!r}, three_dim_layout={self.three_dim_layout!r}, "
            f"final_step_only={self.final_step_only}, exposure_normalize={self.exposure_normalize}, "
            f"min_factor_sigma={self.min_factor_sigma}, min_residual_sigma={self.min_residual_sigma})"
        )

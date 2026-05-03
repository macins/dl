from __future__ import annotations

import math

from ..base import BaseHead
from .registry import register_head
from ...utils.externals import ensure_torch

torch = ensure_torch()
nn = torch.nn
F = torch.nn.functional


def _inverse_softplus(x: float) -> float:
    x = float(x)
    if x <= 0.0:
        x = 1.0
    # numerically stable enough for init.
    return math.log(math.expm1(x))
    
def _as_component_scale_init(
    scale_init: float | list[float] | tuple[float, ...],
    *,
    output_dim: int,
    num_components: int,
    device=None,
    dtype=None,
) -> torch.Tensor:
    if isinstance(scale_init, (int, float)):
        values = torch.full(
            (output_dim, num_components),
            float(scale_init),
            device=device,
            dtype=dtype,
        )
    else:
        values = torch.as_tensor(
            list(scale_init),
            device=device,
            dtype=dtype,
        )

        if values.ndim != 1:
            raise ValueError(
                f"scale_init must be scalar or 1D list/tuple, got shape {tuple(values.shape)}."
            )

        if values.numel() != num_components:
            raise ValueError(
                f"scale_init has {values.numel()} values, but num_components={num_components}."
            )

        values = values.view(1, num_components).expand(output_dim, num_components)

    if torch.any(values <= 0):
        raise ValueError(f"All scale_init values must be positive, got {values}.")

    return values

@register_head("sequence_regression")
class SequenceRegressionHead(BaseHead):
    """
    Old compatible regression head.

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


@register_head("increment_path")
class IncrementPathHead(BaseHead):
    def __init__(
        self,
        *,
        input_dim: int,
        target_key: str = "ret_30min",
        num_horizons: int = 6,
        use_layer_norm: bool = True,
        path_key: str = "path",
        **kwargs, 
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.target_key = str(target_key)
        self.num_horizons = int(num_horizons)
        self.path_key = str(path_key)

        if self.num_horizons <= 0:
            raise ValueError(f"num_horizons must be positive, got {self.num_horizons}")

        self.norm = nn.LayerNorm(self.input_dim) if use_layer_norm else nn.Identity()
        self.proj = nn.Linear(self.input_dim, self.num_horizons, bias=False)

    def forward(self, x: torch.Tensor) -> dict:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"IncrementPathHead expected last dim={self.input_dim}, "
                f"got {x.shape[-1]} for input shape {tuple(x.shape)}."
            )

        pred_inc = self.proj(self.norm(x))
        pred_cum = torch.cumsum(pred_inc, dim=-1)
        pred_30 = pred_cum[..., -1]

        return {
            "preds": {self.target_key: pred_30},
            self.path_key: {"pred_inc": pred_inc, "pred_cum": pred_cum},
        }


@register_head("sequence_mog_regression")
class SequenceMoGRegressionHead(BaseHead):
    """
    Mixture-of-Gaussian sequence regression head.

    Default mode:
        shared_mean=True

        p(y | x) = sum_k pi_k(x) Normal(y; signal(x), sigma_k(x)^2)

    This is the recommended first version for IC-oriented return prediction:
        - preds[target_key] is still the main signal.
        - MoG only models residual scale / tail / regime.

    Full MoG mode:
        shared_mean=False

        p(y | x) = sum_k pi_k(x) Normal(y; mu_k(x), sigma_k(x)^2)

        preds[target_key] becomes the mixture mean:
            sum_k pi_k * mu_k

    Input:
        x: (..., input_dim)

    Output:
        {
            "preds": {
                target_key: (..., output_dim)
            },
            "mog": {
                target_key: {
                    "mix_logits": (..., output_dim, K),
                    "raw_scales": (..., output_dim, K),
                    optional "component_means": (..., output_dim, K)
                }
            }
        }

    Notes:
        - raw_scales are transformed by the objective using softplus + sigma_floor.
        - output_dim=1 is the common case.
        - output units are normalized target units, same as your old head.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        target_key: str = "ret_30min",
        output_dim: int = 1,
        num_components: int = 3,
        use_layer_norm: bool = True,
        shared_mean: bool = False,
        component_mean_scale: float = 0.0,
        scale_init: float | list[float] | tuple[float, ...] = 1.0,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.target_key = str(target_key)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_components = int(num_components)
        self.shared_mean = bool(shared_mean)
        self.component_mean_scale = float(component_mean_scale)

        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {self.output_dim}")

        if self.num_components <= 0:
            raise ValueError(f"num_components must be positive, got {self.num_components}")

        self.norm = nn.LayerNorm(self.input_dim) if use_layer_norm else nn.Identity()

        self.signal_proj = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.mix_proj = nn.Linear(
            self.input_dim,
            self.output_dim * self.num_components,
            bias=bias,
        )
        self.raw_scale_proj = nn.Linear(
            self.input_dim,
            self.output_dim * self.num_components,
            bias=bias,
        )

        if not self.shared_mean:
            self.component_mean_proj = nn.Linear(
                self.input_dim,
                self.output_dim * self.num_components,
                bias=bias,
            )
        else:
            self.component_mean_proj = None

        self._init_parameters(scale_init=scale_init)

    def _init_parameters(
        self,
        *,
        scale_init: float | list[float] | tuple[float, ...],
    ) -> None:
        # Start router close to uniform.
        nn.init.zeros_(self.mix_proj.weight)
        if self.mix_proj.bias is not None:
            nn.init.zeros_(self.mix_proj.bias)
    
        # Start scales with potentially different initial values per component.
        #
        # raw_scale_proj outputs shape:
        #   (..., output_dim * num_components)
        #
        # objective later uses:
        #   sigma = softplus(raw_scales) + sigma_floor
        #
        # So this initializes softplus(raw_scale) ~= scale_init.
        nn.init.zeros_(self.raw_scale_proj.weight)
    
        if self.raw_scale_proj.bias is not None:
            scale_values = _as_component_scale_init(
                scale_init,
                output_dim=self.output_dim,
                num_components=self.num_components,
                device=self.raw_scale_proj.bias.device,
                dtype=self.raw_scale_proj.bias.dtype,
            )
    
            raw_bias = torch.log(torch.expm1(scale_values))
            self.raw_scale_proj.bias.data.copy_(raw_bias.reshape(-1))
    
        # Full-MoG component mean offset starts small.
        if self.component_mean_proj is not None:
            nn.init.zeros_(self.component_mean_proj.weight)
            if self.component_mean_proj.bias is not None:
                nn.init.zeros_(self.component_mean_proj.bias)

    def _reshape_components(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*x.shape[:-1], self.output_dim, self.num_components)

    def forward(self, x: torch.Tensor) -> dict:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"SequenceMoGRegressionHead expected last dim={self.input_dim}, "
                f"got {x.shape[-1]} for input shape {tuple(x.shape)}."
            )

        z = self.norm(x)

        signal = self.signal_proj(z)  # (..., output_dim)

        mix_logits = self._reshape_components(self.mix_proj(z))
        raw_scales = self._reshape_components(self.raw_scale_proj(z))

        mog_params = {
            "mix_logits": mix_logits,
            "raw_scales": raw_scales,
        }

        if self.shared_mean:
            pred = signal
        else:
            raw_component_means = self._reshape_components(self.component_mean_proj(z))

            if self.component_mean_scale > 0.0:
                component_means = (
                    signal.unsqueeze(-1)
                    + self.component_mean_scale * torch.tanh(raw_component_means)
                )
            else:
                component_means = raw_component_means

            pi = torch.softmax(mix_logits, dim=-1)
            pred = torch.sum(pi * component_means, dim=-1)

            mog_params["component_means"] = component_means

        return {
            "preds": {self.target_key: pred},
            "mog": {self.target_key: mog_params},
        }

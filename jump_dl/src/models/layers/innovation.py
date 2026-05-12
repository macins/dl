from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ...utils.externals import ensure_torch

torch = ensure_torch()
nn = torch.nn


@dataclass
class InnovationConfig:
    enabled: bool = False
    latent_dim: int | None = None
    prior_type: str = "gru"
    fusion_type: str = "mlp"
    use_standardized_innovation: bool = True
    aux_loss_weight: float = 0.01
    min_log_s: float = -6.0
    max_log_s: float = 4.0
    detach_aux_target: bool = True
    use_market_product_decomposition: bool = False
    eps: float = 1e-6


class CausalTemporalPrior(nn.Module):
    """Causal GRU prior over time.

    The input must be shift-right latent tokens so that output at time t only
    depends on z_{<t}. This enforces no-leakage for innovation prediction.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)
        self.obs_head = nn.Linear(d_model, d_model * 2)

    def forward(self, z_shifted: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, _ = self.gru(z_shifted)
        out = self.obs_head(h)
        z_pred, log_s = torch.chunk(out, 2, dim=-1)
        return z_pred, log_s


class InnovationTokenAdapter(nn.Module):
    def __init__(self, d_model: int, cfg: InnovationConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_model = int(cfg.latent_dim or d_model)
        if self.d_model != d_model:
            self.in_proj = nn.Linear(d_model, self.d_model)
            self.out_proj = nn.Linear(self.d_model, d_model)
        else:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()

        if cfg.prior_type != "gru":
            raise ValueError(f"Unsupported innovation prior_type={cfg.prior_type!r}; only 'gru' is implemented.")
        self.prior = CausalTemporalPrior(self.d_model)
        self.start_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        fusion_in = self.d_model * 4 + 1
        if not cfg.use_standardized_innovation:
            fusion_in = self.d_model * 3 + 1
        self.fuse = nn.Sequential(nn.Linear(fusion_in, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def _shift_right(self, z: torch.Tensor) -> torch.Tensor:
        z_shifted = torch.zeros_like(z)
        z_shifted[:, 1:, :] = z[:, :-1, :]
        z_shifted[:, 0:1, :] = self.start_token
        return z_shifted

    def forward(self, z: torch.Tensor, product_ids: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> dict:
        del product_ids  # TODO: optional market/product/idiosyncratic decomposition
        orig_shape = z.shape
        if z.ndim == 4:
            b, n, t, d = z.shape
            z_work = z.reshape(b * n, t, d)
            mask_work = None if mask is None else mask.reshape(b * n, t)
        elif z.ndim == 3:
            b, t, d = z.shape
            z_work = z
            mask_work = mask
        else:
            raise ValueError(f"InnovationTokenAdapter expects z with shape (B,T,D) or (B,N,T,D), got {tuple(z.shape)}")

        z_latent = self.in_proj(z_work)
        z_shifted = self._shift_right(z_latent)
        z_pred, log_s = self.prior(z_shifted)
        log_s = torch.clamp(log_s, min=self.cfg.min_log_s, max=self.cfg.max_log_s)

        z_target = z_latent.detach() if self.cfg.detach_aux_target else z_latent
        innovation = z_latent - z_pred
        scale = torch.exp(log_s).clamp_min(self.cfg.eps)
        innovation_std = innovation / (scale + self.cfg.eps)

        aux_nll = 0.5 * ((z_target - z_pred) / (scale + self.cfg.eps)).pow(2) + log_s
        valid = torch.ones_like(aux_nll[..., 0], dtype=torch.bool)
        valid[:, 0] = False
        if mask_work is not None:
            valid = valid & mask_work.bool()
        valid_f = valid.unsqueeze(-1).float()
        denom = valid_f.sum().clamp_min(1.0)
        aux_loss = (aux_nll * valid_f).sum() / denom

        log_s_mean = float(log_s.detach().mean().item())
        residual = (z_latent - z_pred).detach()
        var_z = torch.var(z_latent.detach(), unbiased=False)
        var_res = torch.var(residual, unbiased=False)
        predictability_r2 = float((1.0 - (var_res / (var_z + self.cfg.eps))).item())

        log_s_expanded = log_s.mean(dim=-1, keepdim=True)
        fuse_inputs = [z_latent, z_pred, innovation]
        if self.cfg.use_standardized_innovation:
            fuse_inputs.append(innovation_std)
        fuse_inputs.append(log_s_expanded)
        token = self.fuse(torch.cat(fuse_inputs, dim=-1))
        token = self.out_proj(token)

        if len(orig_shape) == 4:
            token = token.reshape(orig_shape[0], orig_shape[1], orig_shape[2], -1)
            z_pred = z_pred.reshape(orig_shape[0], orig_shape[1], orig_shape[2], -1)
            log_s = log_s.reshape(orig_shape[0], orig_shape[1], orig_shape[2], -1)
            innovation = innovation.reshape(orig_shape[0], orig_shape[1], orig_shape[2], -1)
            innovation_std = innovation_std.reshape(orig_shape[0], orig_shape[1], orig_shape[2], -1)

        return {
            "token": token,
            "z": z,
            "z_pred": z_pred,
            "log_s": log_s,
            "innovation": innovation,
            "innovation_std": innovation_std,
            "aux_loss": aux_loss,
            "diagnostics": {
                "innovation/aux_nll": float(aux_loss.detach().item()),
                "innovation/z_pred_mse": float(torch.mean((z_latent.detach() - z_pred.detach()) ** 2).item()),
                "innovation/mean_abs_innovation": float(innovation.detach().abs().mean().item()),
                "innovation/mean_abs_innovation_std": float(innovation_std.detach().abs().mean().item()),
                "innovation/log_s_mean": log_s_mean,
                "innovation/log_s_min": float(log_s.detach().min().item()),
                "innovation/log_s_max": float(log_s.detach().max().item()),
                "innovation/predictability_r2": predictability_r2,
            },
        }

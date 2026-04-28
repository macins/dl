from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .metrics import masked_cosine_similarity
from .utils.externals import ensure_torch

torch = ensure_torch()
nn = torch.nn
F = torch.nn.functional


@dataclass
class ObjectiveOutput:
    loss: torch.Tensor
    metrics: dict[str, float]


class CosineSimilarityObjective(nn.Module):
    """
    Backward-compatible objective with optional MoG NLL.

    Old usage still works:

        objective:
          name: cosine_similarity
          lam_cos: 1.0
          lam_mse: 1.0

    New weighted usage:

        objective:
          name: cosine_similarity
          pred_key: ret_30min
          target_key: ret_30min
          target_mean: 0.0
          target_std: 1.0

          loss_weights:
            cos: 1.0
            mse: 0.0
            mog_nll: 0.05
            usage_kl: 0.001
            scale_reg: 0.0
            aux: 1.0

          loss_schedules:
            cos:
              by: step
              mode: cosine
              start_step: 0
              end_step: 10000
              start_weight: 1.0
              end_weight: 0.3
            mse:
              by: step
              mode: cosine
              start_step: 0
              end_step: 10000
              start_weight: 0.0
              end_weight: 1.0
            mog_nll:
              by: step
              mode: linear
              start_step: 1000
              end_step: 10000
              start_weight: 0.0
              end_weight: 0.05

    Loss definitions:
        cos:
            loss_cos = -cosine_similarity(pred_raw, target_raw)

        mse:
            loss_mse = mean((pred_norm - target_norm)^2)

        mog_nll:
            If head is shared-mean MoG:
                residual = target_norm - pred_norm
                p(residual | x) = sum_k pi_k Normal(residual; 0, sigma_k^2)

            If head outputs component_means:
                p(target_norm | x) = sum_k pi_k Normal(target_norm; mu_k, sigma_k^2)

        usage_kl:
            KL(mean_batch(pi) || uniform), penalizes component collapse.

        scale_reg:
            mean(log(sigma)^2), mild optional regularizer.

    Important:
        - MoG NLL is computed in normalized target units.
        - Cosine/mse_raw metrics remain in raw target units where appropriate.
        - The trainer can call set_training_progress(...) before every step.
    """

    def __init__(
        self,
        lam_cos: float = 1.0,
        lam_mse: float = 1.0,
        pred_key: str = "ret_30min",
        target_key: str = "ret_30min",
        target_mean: float = 0.0,
        target_std: float = 1.0,
        *,
        pred_index: int | None = None,
        target_index: int | None = None,
        # New optional MoG / weighted-loss config.
        lam_mog_nll: float = 0.0,
        lam_usage_kl: float = 0.0,
        lam_scale_reg: float = 0.0,
        aux_loss_weight: float = 1.0,
        loss_weights: Mapping[str, float] | None = None,
        loss_schedules: Mapping[str, Mapping[str, Any]] | None = None,
        mog_key: str = "mog",
        sigma_floor: float = 1e-4,
        sigma_max: float | None = None,
        use_sample_weight: bool = False,
        weight_key: str = "weight",
        eps: float = 1e-12,
    ) -> None:
        super().__init__()

        self.pred_key = str(pred_key)
        self.target_key = str(target_key)

        self.target_mean = float(target_mean)
        self.target_std = float(target_std) if float(target_std) != 0.0 else 1.0

        self.pred_index = pred_index
        self.target_index = target_index

        self.mog_key = str(mog_key)
        self.sigma_floor = float(sigma_floor)
        self.sigma_max = None if sigma_max is None else float(sigma_max)

        self.use_sample_weight = bool(use_sample_weight)
        self.weight_key = str(weight_key)
        self.eps = float(eps)

        self.loss_weights: dict[str, float] = {
            "cos": float(lam_cos),
            "mse": float(lam_mse),
            "mog_nll": float(lam_mog_nll),
            "usage_kl": float(lam_usage_kl),
            "scale_reg": float(lam_scale_reg),
            "aux": float(aux_loss_weight),
        }

        if loss_weights is not None:
            for key, value in loss_weights.items():
                self.loss_weights[str(key)] = float(value)

        self.loss_schedules: dict[str, dict[str, Any]] = {}
        if loss_schedules is not None:
            for key, value in loss_schedules.items():
                self.loss_schedules[str(key)] = dict(value)

        self.global_step = 0
        self.current_epoch = 0
        self.num_epochs = 1
        self.training_context = True

    # ------------------------------------------------------------------
    # Trainer hook for scheduled loss weights
    # ------------------------------------------------------------------

    def set_training_progress(
        self,
        *,
        global_step: int,
        epoch: int,
        num_epochs: int,
        train: bool,
    ) -> None:
        self.global_step = int(global_step)
        self.current_epoch = int(epoch)
        self.num_epochs = max(int(num_epochs), 1)
        self.training_context = bool(train)

    def _scheduled_weight(self, name: str) -> float:
        base = float(self.loss_weights.get(name, 0.0))
        cfg = self.loss_schedules.get(name)

        if cfg is None:
            return base

        by = str(cfg.get("by", cfg.get("schedule_by", "step"))).lower()

        if by in {"step", "global_step", "steps"}:
            current = float(self.global_step)
            start = float(cfg.get("start", cfg.get("start_step", 0)))
            end = float(cfg.get("end", cfg.get("end_step", start)))
        elif by in {"epoch", "epochs"}:
            current = float(self.current_epoch)
            start = float(cfg.get("start", cfg.get("start_epoch", 1)))
            end = float(cfg.get("end", cfg.get("end_epoch", self.num_epochs)))
        else:
            raise ValueError(f"Unknown loss schedule unit {by!r} for loss {name!r}")

        start_weight = float(cfg.get("start_weight", base))
        end_weight = float(cfg.get("end_weight", base))

        if end <= start:
            return end_weight if current >= end else start_weight

        t = (current - start) / (end - start)
        t = max(0.0, min(1.0, t))

        mode = str(cfg.get("mode", cfg.get("type", "linear"))).lower()

        if mode in {"constant", "const"}:
            alpha = 0.0
        elif mode in {"linear", "lin"}:
            alpha = t
        elif mode in {"cosine", "cos"}:
            alpha = 0.5 - 0.5 * math.cos(math.pi * t)
        elif mode in {"step"}:
            alpha = 0.0 if t < 1.0 else 1.0
        else:
            raise ValueError(f"Unknown loss schedule mode {mode!r} for loss {name!r}")

        return (1.0 - alpha) * start_weight + alpha * end_weight

    def get_current_loss_weights(self) -> dict[str, float]:
        return {name: self._scheduled_weight(name) for name in self.loss_weights}

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def unnormalize_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        return y_pred * self.target_std + self.target_mean

    def normalize_target(self, y_true: torch.Tensor) -> torch.Tensor:
        return (y_true - self.target_mean) / self.target_std

    # ------------------------------------------------------------------
    # Tensor selection utilities
    # ------------------------------------------------------------------

    def _select_from_mapping(self, value: Mapping[str, Any], key: str) -> torch.Tensor:
        if key not in value:
            raise KeyError(f"Key {key!r} was not found. Available keys: {sorted(value.keys())}")

        selected = value[key]
        if not torch.is_tensor(selected):
            raise TypeError(f"Expected tensor for key {key!r}, got {type(selected)!r}")

        return selected

    def _select_from_tensor(
        self,
        value: torch.Tensor,
        index: int | None,
        kind: str,
    ) -> torch.Tensor:
        if value.ndim == 0:
            raise ValueError(f"{kind} tensor must have at least one dimension.")

        # Common scalar-per-token shapes:
        #   old:   (B, T)
        #   panel: (B, N, T)
        #
        # This also means that true multi-target tensors with shape (B, T, C)
        # should pass index explicitly if ambiguity matters.
        if value.ndim <= 3:
            if index is None:
                return value
            return value[..., int(index)]

        # If last dim is singleton target dim:
        #   (B, T, 1)
        #   (B, N, T, 1)
        if value.shape[-1] == 1 and index is None:
            return value[..., 0]

        if index is None:
            raise ValueError(
                f"{kind} tensor has shape {tuple(value.shape)}. "
                f"Please specify {kind}_index when using multi-target tensors."
            )

        return value[..., int(index)]

    def get_prediction_tensor(self, outputs: dict) -> torch.Tensor:
        preds = outputs["preds"]

        if isinstance(preds, Mapping):
            return self._select_from_tensor(
                self._select_from_mapping(preds, self.pred_key),
                self.pred_index,
                "pred",
            )

        if torch.is_tensor(preds):
            return self._select_from_tensor(preds, self.pred_index, "pred")

        raise TypeError(f"Unsupported preds container: {type(preds)!r}")

    def get_target_tensor(self, batch: dict) -> torch.Tensor:
        targets = batch["targets"]

        if isinstance(targets, Mapping):
            return self._select_from_tensor(
                self._select_from_mapping(targets, self.target_key),
                self.target_index,
                "target",
            )

        if torch.is_tensor(targets):
            return self._select_from_tensor(targets, self.target_index, "target")

        raise TypeError(f"Unsupported targets container: {type(targets)!r}")

    def get_mask_tensor(self, batch: dict) -> torch.Tensor:
        """
        Mask priority:

        1. batch["loss_mask"]
        2. targets["valid_mask"]
        3. batch["padding_mask"]
        """
        if "loss_mask" in batch:
            mask = batch["loss_mask"]
        else:
            targets = batch.get("targets")
            if isinstance(targets, Mapping) and "valid_mask" in targets:
                mask = targets["valid_mask"]
            else:
                mask = batch.get("padding_mask")

        if mask is None or not torch.is_tensor(mask):
            raise KeyError(
                "Batch does not contain a valid mask. Expected batch['loss_mask'], "
                "targets['valid_mask'], or batch['padding_mask']."
            )

        return mask.bool()

    def get_sample_weight_tensor(
        self,
        batch: dict,
        *,
        y_ref: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.use_sample_weight:
            return None

        weight = None

        targets = batch.get("targets")
        if isinstance(targets, Mapping) and self.weight_key in targets:
            weight = targets[self.weight_key]
        elif self.weight_key in batch:
            weight = batch[self.weight_key]
        elif "sample_weight" in batch:
            weight = batch["sample_weight"]

        if weight is None:
            return None

        if not torch.is_tensor(weight):
            raise TypeError(f"Sample weight must be a tensor, got {type(weight)!r}")

        weight = self._select_from_tensor(weight, self.target_index, "sample_weight")

        if weight.shape != y_ref.shape:
            raise ValueError(
                f"Sample weight shape mismatch: weight={tuple(weight.shape)}, "
                f"target={tuple(y_ref.shape)}."
            )

        return weight

    @staticmethod
    def _validate_shapes(
        *,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Prediction/target shape mismatch: "
                f"pred={tuple(y_pred.shape)}, target={tuple(y_true.shape)}."
            )

        if mask.shape != y_true.shape:
            raise ValueError(
                f"Mask/target shape mismatch: "
                f"mask={tuple(mask.shape)}, target={tuple(y_true.shape)}. "
                "After target selection, mask should match target shape."
            )

    # ------------------------------------------------------------------
    # Weighted helpers
    # ------------------------------------------------------------------

    def _make_loss_weight_tensor(
        self,
        *,
        valid: torch.Tensor,
        sample_weight: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if sample_weight is None:
            w = valid.to(dtype=dtype)
        else:
            w = sample_weight.to(dtype=dtype)
            w = torch.where(valid, w, torch.zeros_like(w))
            w = torch.where(torch.isfinite(w), w, torch.zeros_like(w))
            w = torch.clamp(w, min=0.0)

        return w

    def _weighted_mean(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        denom = torch.sum(w).clamp_min(self.eps)
        return torch.sum(x * w) / denom

    def _weighted_cosine_similarity(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        w: torch.Tensor,
    ) -> torch.Tensor:
        y_pred = y_pred.float()
        y_true = y_true.float()
        w = w.float()

        numerator = torch.sum(w * y_pred * y_true)
        pred_norm = torch.sum(w * y_pred * y_pred).clamp_min(self.eps).sqrt()
        true_norm = torch.sum(w * y_true * y_true).clamp_min(self.eps).sqrt()

        return numerator / (pred_norm * true_norm).clamp_min(self.eps)

    # ------------------------------------------------------------------
    # Aux losses
    # ------------------------------------------------------------------

    def get_aux_losses(self, outputs: dict) -> tuple[torch.Tensor, dict[str, float]]:
        aux_losses = outputs.get("aux_losses", {})
        aux_metrics = outputs.get("aux_metrics", {})

        preds = self.get_prediction_tensor(outputs)
        total = torch.tensor(0.0, device=preds.device, dtype=torch.float32)

        metrics: dict[str, float] = {}

        if isinstance(aux_losses, Mapping):
            for name, value in aux_losses.items():
                if torch.is_tensor(value):
                    value = value.float()
                    total = total + value
                    metrics[name] = float(value.detach().item())

        if isinstance(aux_metrics, Mapping):
            for name, value in aux_metrics.items():
                if isinstance(value, (int, float)):
                    metrics[name] = float(value)

        return total, metrics

    # ------------------------------------------------------------------
    # MoG utilities
    # ------------------------------------------------------------------

    def _get_mog_container(self, outputs: dict) -> Mapping[str, Any] | None:
        mog = outputs.get(self.mog_key)
        if mog is None:
            return None

        if not isinstance(mog, Mapping):
            raise TypeError(f"outputs[{self.mog_key!r}] must be a mapping, got {type(mog)!r}")

        if self.pred_key in mog:
            container = mog[self.pred_key]
        elif self.target_key in mog:
            container = mog[self.target_key]
        else:
            # Allow flat outputs["mog"] = {"mix_logits": ..., "raw_scales": ...}
            container = mog

        if not isinstance(container, Mapping):
            raise TypeError(
                f"MoG container for key {self.pred_key!r}/{self.target_key!r} "
                f"must be a mapping, got {type(container)!r}"
            )

        return container

    def _select_component_param(
        self,
        value: torch.Tensor,
        *,
        y_ref: torch.Tensor,
        index: int | None,
        kind: str,
    ) -> torch.Tensor:
        """
        Converts:
            (..., K)             -> (..., K)
            (..., 1, K)          -> (..., K)
            (..., output_dim, K) -> (..., K), if index is specified
        where ... must match y_ref.shape.
        """
        if value.ndim < 2:
            raise ValueError(f"{kind} must have at least 2 dims, got {tuple(value.shape)}")

        if tuple(value.shape[:-1]) == tuple(y_ref.shape):
            return value

        if tuple(value.shape[:-2]) == tuple(y_ref.shape):
            if value.shape[-2] == 1 and index is None:
                return value[..., 0, :]

            if index is None:
                raise ValueError(
                    f"{kind} has shape {tuple(value.shape)}. "
                    f"Please specify target_index/pred_index for multi-target MoG."
                )

            return value[..., int(index), :]

        raise ValueError(
            f"{kind} shape mismatch. Expected prefix {tuple(y_ref.shape)} with final K, "
            f"or prefix {tuple(y_ref.shape)} with final output_dim,K. "
            f"Got {tuple(value.shape)}."
        )

    def _get_mog_tensor(
        self,
        container: Mapping[str, Any],
        name: str,
        *,
        y_ref: torch.Tensor,
        required: bool,
    ) -> torch.Tensor | None:
        if name not in container:
            if required:
                raise KeyError(
                    f"MoG output missing key {name!r}. "
                    f"Available keys: {sorted(container.keys())}"
                )
            return None

        value = container[name]
        if not torch.is_tensor(value):
            raise TypeError(f"MoG key {name!r} must be tensor, got {type(value)!r}")

        return self._select_component_param(
            value,
            y_ref=y_ref,
            index=self.target_index,
            kind=name,
        )

    def _compute_mog_terms(
        self,
        *,
        outputs: dict,
        y_pred_norm: torch.Tensor,
        y_true_norm: torch.Tensor,
        valid: torch.Tensor,
        w: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        container = self._get_mog_container(outputs)
        if container is None:
            zero = torch.tensor(0.0, device=y_pred_norm.device, dtype=torch.float32)
            return {
                "mog_nll": zero,
                "usage_kl": zero,
                "scale_reg": zero,
            }, {}

        mix_logits = self._get_mog_tensor(
            container,
            "mix_logits",
            y_ref=y_true_norm,
            required=True,
        )
        raw_scales = self._get_mog_tensor(
            container,
            "raw_scales",
            y_ref=y_true_norm,
            required=False,
        )

        if raw_scales is None:
            # Also support old naming if you later choose to output log_scales.
            raw_scales = self._get_mog_tensor(
                container,
                "log_scales",
                y_ref=y_true_norm,
                required=True,
            )

        component_means = self._get_mog_tensor(
            container,
            "component_means",
            y_ref=y_true_norm,
            required=False,
        )

        mix_logits = mix_logits.float()
        raw_scales = raw_scales.float()
        y_true_norm = y_true_norm.float()
        y_pred_norm = y_pred_norm.float()
        w = w.float()

        if mix_logits.shape[:-1] != y_true_norm.shape:
            raise ValueError(
                f"mix_logits shape mismatch: logits={tuple(mix_logits.shape)}, "
                f"target={tuple(y_true_norm.shape)}."
            )

        if raw_scales.shape != mix_logits.shape:
            raise ValueError(
                f"raw_scales/mix_logits shape mismatch: "
                f"raw_scales={tuple(raw_scales.shape)}, logits={tuple(mix_logits.shape)}."
            )

        sigma = F.softplus(raw_scales) + self.sigma_floor
        if self.sigma_max is not None:
            sigma = sigma.clamp(max=self.sigma_max)

        log_sigma = torch.log(sigma.clamp_min(self.eps))
        log_pi = torch.log_softmax(mix_logits, dim=-1)

        if component_means is None:
            loc = y_pred_norm.unsqueeze(-1)
        else:
            component_means = component_means.float()
            if component_means.shape != mix_logits.shape:
                raise ValueError(
                    f"component_means/mix_logits shape mismatch: "
                    f"component_means={tuple(component_means.shape)}, "
                    f"logits={tuple(mix_logits.shape)}."
                )
            loc = component_means

        residual = y_true_norm.unsqueeze(-1) - loc

        log_prob_k = (
            log_pi
            - 0.5 * (residual / sigma).pow(2)
            - log_sigma
            - 0.5 * math.log(2.0 * math.pi)
        )

        nll_per_token = -torch.logsumexp(log_prob_k, dim=-1)

        if valid.any():
            mog_nll = self._weighted_mean(nll_per_token, w)

            pi = torch.softmax(mix_logits, dim=-1)
            denom = torch.sum(w).clamp_min(self.eps)
            pi_bar = torch.sum(pi * w.unsqueeze(-1), dim=tuple(range(w.ndim))) / denom
            pi_bar = pi_bar.clamp_min(self.eps)
            K = pi_bar.numel()

            usage_kl = torch.sum(pi_bar * (torch.log(pi_bar) + math.log(float(K))))

            scale_reg_per_token = log_sigma.pow(2).mean(dim=-1)
            scale_reg = self._weighted_mean(scale_reg_per_token, w)

            sigma_mean = self._weighted_mean(sigma.mean(dim=-1), w)
            sigma_min = torch.min(sigma.detach())
            sigma_max = torch.max(sigma.detach())
            usage_entropy = -torch.sum(pi_bar * torch.log(pi_bar))
        else:
            mog_nll = torch.tensor(0.0, device=y_pred_norm.device, dtype=torch.float32)
            usage_kl = torch.tensor(0.0, device=y_pred_norm.device, dtype=torch.float32)
            scale_reg = torch.tensor(0.0, device=y_pred_norm.device, dtype=torch.float32)
            sigma_mean = torch.tensor(0.0, device=y_pred_norm.device, dtype=torch.float32)
            sigma_min = torch.tensor(0.0, device=y_pred_norm.device, dtype=torch.float32)
            sigma_max = torch.tensor(0.0, device=y_pred_norm.device, dtype=torch.float32)
            usage_entropy = torch.tensor(0.0, device=y_pred_norm.device, dtype=torch.float32)

        terms = {
            "mog_nll": mog_nll,
            "usage_kl": usage_kl,
            "scale_reg": scale_reg,
        }

        metrics = {
            "mog_nll": float(mog_nll.detach().item()),
            "mog_usage_kl": float(usage_kl.detach().item()),
            "mog_usage_entropy": float(usage_entropy.detach().item()),
            "mog_sigma_mean": float(sigma_mean.detach().item()),
            "mog_sigma_min": float(sigma_min.detach().item()),
            "mog_sigma_max": float(sigma_max.detach().item()),
        }

        return terms, metrics

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, outputs: dict, batch: dict) -> ObjectiveOutput:
        y_pred = self.get_prediction_tensor(outputs)
        y_true = self.get_target_tensor(batch)
        mask = self.get_mask_tensor(batch)

        self._validate_shapes(y_pred=y_pred, y_true=y_true, mask=mask)

        y_pred = y_pred.float()
        y_true = y_true.float()
        valid = mask.bool()

        sample_weight = self.get_sample_weight_tensor(batch, y_ref=y_true)
        w = self._make_loss_weight_tensor(
            valid=valid,
            sample_weight=sample_weight,
            dtype=torch.float32,
        )

        y_pred_unnorm = self.unnormalize_prediction(y_pred)
        y_true_norm = self.normalize_target(y_true)

        if valid.any():
            if sample_weight is None:
                cos = masked_cosine_similarity(y_pred_unnorm, y_true, valid)
            else:
                cos = self._weighted_cosine_similarity(y_pred_unnorm, y_true, w)

            mse_raw = self._weighted_mean((y_pred_unnorm - y_true).pow(2), w)
            mse_normalized = self._weighted_mean((y_pred - y_true_norm).pow(2), w)
        else:
            cos = torch.tensor(0.0, device=y_pred.device, dtype=torch.float32)
            mse_raw = torch.tensor(0.0, device=y_pred.device, dtype=torch.float32)
            mse_normalized = torch.tensor(0.0, device=y_pred.device, dtype=torch.float32)

        mog_terms, mog_metrics = self._compute_mog_terms(
            outputs=outputs,
            y_pred_norm=y_pred,
            y_true_norm=y_true_norm,
            valid=valid,
            w=w,
        )

        aux_loss_total, aux_metrics = self.get_aux_losses(outputs)

        weights = self.get_current_loss_weights()

        loss_cos = -cos
        loss_mse = mse_normalized
        loss_mog_nll = mog_terms["mog_nll"]
        loss_usage_kl = mog_terms["usage_kl"]
        loss_scale_reg = mog_terms["scale_reg"]

        loss = (
            weights.get("cos", 0.0) * loss_cos
            + weights.get("mse", 0.0) * loss_mse
            + weights.get("mog_nll", 1.0) * loss_mog_nll
            + weights.get("usage_kl", 0.0) * loss_usage_kl
            + weights.get("scale_reg", 0.0) * loss_scale_reg
            + weights.get("aux", 1.0) * aux_loss_total
        )

        metrics = {
            "cosine_similarity": float(cos.detach().item()),
            "mse": float(mse_normalized.detach().item()),
            "mse_raw": float(mse_raw.detach().item()),
            "loss_cos": float(loss_cos.detach().item()),
            "loss_mse": float(loss_mse.detach().item()),
            "loss_mog_nll": float(loss_mog_nll.detach().item()),
            "loss_usage_kl": float(loss_usage_kl.detach().item()),
            "loss_scale_reg": float(loss_scale_reg.detach().item()),
            "aux_loss": float(aux_loss_total.detach().item()),
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            **{f"loss_weight_{name}": float(value) for name, value in weights.items()},
            **mog_metrics,
            **aux_metrics,
        }

        return ObjectiveOutput(loss=loss, metrics=metrics)
    def __repr__(self) -> str:
        def fmt_float(x: float | None) -> str:
            if x is None:
                return "None"
            x = float(x)
            if x == 0.0:
                return "0.0"
            if abs(x) < 1e-3 or abs(x) >= 1e4:
                return f"{x:.3e}"
            return f"{x:.6g}"
    
        def fmt_mapping(m: Mapping[str, Any]) -> str:
            if not m:
                return "{}"
    
            parts = []
            for k, v in m.items():
                if isinstance(v, float):
                    parts.append(f"{k}={fmt_float(v)}")
                else:
                    parts.append(f"{k}={v!r}")
    
            return "{" + ", ".join(parts) + "}"
    
        def fmt_schedule(name: str, cfg: Mapping[str, Any]) -> str:
            by = cfg.get("by", cfg.get("schedule_by", "step"))
            mode = cfg.get("mode", cfg.get("type", "linear"))
    
            if str(by).lower() in {"step", "global_step", "steps"}:
                start = cfg.get("start", cfg.get("start_step", 0))
                end = cfg.get("end", cfg.get("end_step", start))
                unit = "step"
            else:
                start = cfg.get("start", cfg.get("start_epoch", 1))
                end = cfg.get("end", cfg.get("end_epoch", self.num_epochs))
                unit = "epoch"
    
            base = self.loss_weights.get(name, 0.0)
            start_weight = cfg.get("start_weight", base)
            end_weight = cfg.get("end_weight", base)
    
            return (
                f"{name}: {mode}, by={unit}, "
                f"{start}->{end}, "
                f"{fmt_float(start_weight)}->{fmt_float(end_weight)}"
            )
    
        current_weights = self.get_current_loss_weights()
    
        lines = [
            f"{self.__class__.__name__}(",
            f"  pred_key={self.pred_key!r}, target_key={self.target_key!r},",
            f"  pred_index={self.pred_index!r}, target_index={self.target_index!r},",
            (
                "  target_normalization="
                f"(mean={fmt_float(self.target_mean)}, std={fmt_float(self.target_std)}),"
            ),
            (
                "  mog="
                f"(key={self.mog_key!r}, "
                f"sigma_floor={fmt_float(self.sigma_floor)}, "
                f"sigma_max={fmt_float(self.sigma_max)}),"
            ),
            (
                "  sample_weight="
                f"(enabled={self.use_sample_weight}, weight_key={self.weight_key!r}),"
            ),
            (
                "  progress="
                f"(global_step={self.global_step}, "
                f"epoch={self.current_epoch}/{self.num_epochs}, "
                f"train={self.training_context}),"
            ),
            f"  loss_weights={fmt_mapping(self.loss_weights)},",
            f"  current_loss_weights={fmt_mapping(current_weights)},",
        ]
    
        if self.loss_schedules:
            lines.append("  loss_schedules=[")
            for name, cfg in self.loss_schedules.items():
                lines.append(f"    {fmt_schedule(name, cfg)},")
            lines.append("  ],")
        else:
            lines.append("  loss_schedules=[],")
    
        lines.append(f"  eps={fmt_float(self.eps)}")
        lines.append(")")
    
        return "\n".join(lines)
        
    def __repr__(self) -> str:
        def fmt_float(x: float | None) -> str:
            if x is None:
                return "None"
            x = float(x)
            if x == 0.0:
                return "0.0"
            if abs(x) < 1e-3 or abs(x) >= 1e4:
                return f"{x:.3e}"
            return f"{x:.6g}"
    
        def fmt_mapping(m: Mapping[str, Any]) -> str:
            if not m:
                return "{}"
    
            parts = []
            for k, v in m.items():
                if isinstance(v, float):
                    parts.append(f"{k}={fmt_float(v)}")
                else:
                    parts.append(f"{k}={v!r}")
    
            return "{" + ", ".join(parts) + "}"
    
        def fmt_schedule(name: str, cfg: Mapping[str, Any]) -> str:
            by = cfg.get("by", cfg.get("schedule_by", "step"))
            mode = cfg.get("mode", cfg.get("type", "linear"))
    
            if str(by).lower() in {"step", "global_step", "steps"}:
                start = cfg.get("start", cfg.get("start_step", 0))
                end = cfg.get("end", cfg.get("end_step", start))
                unit = "step"
            else:
                start = cfg.get("start", cfg.get("start_epoch", 1))
                end = cfg.get("end", cfg.get("end_epoch", self.num_epochs))
                unit = "epoch"
    
            base = self.loss_weights.get(name, 0.0)
            start_weight = cfg.get("start_weight", base)
            end_weight = cfg.get("end_weight", base)
    
            return (
                f"{name}: {mode}, by={unit}, "
                f"{start}->{end}, "
                f"{fmt_float(start_weight)}->{fmt_float(end_weight)}"
            )
    
        current_weights = self.get_current_loss_weights()
    
        lines = [
            f"{self.__class__.__name__}(",
            f"  pred_key={self.pred_key!r}, target_key={self.target_key!r},",
            f"  pred_index={self.pred_index!r}, target_index={self.target_index!r},",
            (
                "  target_normalization="
                f"(mean={fmt_float(self.target_mean)}, std={fmt_float(self.target_std)}),"
            ),
            (
                "  mog="
                f"(key={self.mog_key!r}, "
                f"sigma_floor={fmt_float(self.sigma_floor)}, "
                f"sigma_max={fmt_float(self.sigma_max)}),"
            ),
            (
                "  sample_weight="
                f"(enabled={self.use_sample_weight}, weight_key={self.weight_key!r}),"
            ),
            (
                "  progress="
                f"(global_step={self.global_step}, "
                f"epoch={self.current_epoch}/{self.num_epochs}, "
                f"train={self.training_context}),"
            ),
            f"  loss_weights={fmt_mapping(self.loss_weights)},",
            f"  current_loss_weights={fmt_mapping(current_weights)},",
        ]
    
        if self.loss_schedules:
            lines.append("  loss_schedules=[")
            for name, cfg in self.loss_schedules.items():
                lines.append(f"    {fmt_schedule(name, cfg)},")
            lines.append("  ],")
        else:
            lines.append("  loss_schedules=[],")
    
        lines.append(f"  eps={fmt_float(self.eps)}")
        lines.append(")")
    
        return "\n".join(lines)
class MoGRegressionObjective(CosineSimilarityObjective):
    """
    Convenience alias/class.

    Same implementation as CosineSimilarityObjective, but defaults to enabling
    a small MoG NLL weight if loss_weights is not explicitly provided.
    """

    def __init__(
        self,
        *args,
        lam_mog_nll: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, lam_mog_nll=lam_mog_nll, **kwargs)
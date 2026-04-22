from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .metrics import masked_cosine_similarity
from .utils.externals import ensure_torch

torch = ensure_torch()
nn = torch.nn


@dataclass
class ObjectiveOutput:
    loss: torch.Tensor
    metrics: dict[str, float]


class CosineSimilarityObjective(nn.Module):
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
    ) -> None:
        super().__init__()
        self.lam_cos = lam_cos
        self.lam_mse = lam_mse
        self.pred_key = pred_key
        self.target_key = target_key
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        self.pred_index = pred_index
        self.target_index = target_index
        

    def unnormalize_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        return y_pred * self.target_std + self.target_mean

    def normalize_target(self, y_true: torch.Tensor) -> torch.Tensor:
        return (y_true - self.target_mean) / self.target_std

    def _select_from_mapping(self, value: Mapping[str, Any], key: str) -> torch.Tensor:
        if key not in value:
            raise KeyError(f"Key '{key}' was not found. Available keys: {sorted(value.keys())}")
        selected = value[key]
        if not torch.is_tensor(selected):
            raise TypeError(f"Expected tensor for key '{key}', got {type(selected)!r}")
        return selected

    def _select_from_tensor(self, value: torch.Tensor, index: int | None, kind: str) -> torch.Tensor:
        if value.ndim == 0:
            raise ValueError(f"{kind} tensor must have at least one dimension.")
        if value.ndim < 3:
            return value
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
            return self._select_from_tensor(self._select_from_mapping(preds, self.pred_key), self.pred_index, "pred")
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
        targets = batch.get("targets")
        if isinstance(targets, Mapping) and "valid_mask" in targets:
            mask = targets["valid_mask"]
        else:
            mask = batch.get("padding_mask")
        if mask is None or not torch.is_tensor(mask):
            raise KeyError("Batch does not contain a valid mask. Expected 'padding_mask' or targets['valid_mask'].")
        return mask.bool()

    def get_aux_losses(self, outputs: dict) -> tuple[torch.Tensor, dict[str, float]]:
        aux_losses = outputs.get("aux_losses", {})
        aux_metrics = outputs.get("aux_metrics", {})
        preds = self.get_prediction_tensor(outputs)
        total = torch.tensor(0.0, device=preds.device, dtype=preds.dtype)
        metrics: dict[str, float] = {}
        if isinstance(aux_losses, Mapping):
            for name, value in aux_losses.items():
                if torch.is_tensor(value):
                    total = total + value
                    metrics[name] = float(value.detach().item())
        if isinstance(aux_metrics, Mapping):
            for name, value in aux_metrics.items():
                if isinstance(value, (int, float)):
                    metrics[name] = float(value)
        return total, metrics

    def forward(self, outputs: dict, batch: dict) -> ObjectiveOutput:
        y_pred = self.get_prediction_tensor(outputs)
        y_true = self.get_target_tensor(batch)
        mask = self.get_mask_tensor(batch)

        y_pred_unnorm = self.unnormalize_prediction(y_pred)
        y_true_norm = self.normalize_target(y_true)

        cos = masked_cosine_similarity(y_pred_unnorm, y_true, mask)
        mse_raw = torch.mean((y_pred_unnorm[mask] - y_true[mask]) ** 2) if mask.any() else torch.tensor(
            0.0, device=y_pred.device, dtype=y_pred.dtype
        )
        mse_normalized = torch.mean((y_pred[mask] - y_true_norm[mask]) ** 2) if mask.any() else torch.tensor(
            0.0, device=y_pred.device, dtype=y_pred.dtype
        )
        aux_loss_total, aux_metrics = self.get_aux_losses(outputs)
        return ObjectiveOutput(
            loss=self.lam_mse * mse_normalized - self.lam_cos * cos + aux_loss_total,
            metrics={
                "cosine_similarity": float(cos.detach().item()),
                "mse": float(mse_normalized.detach().item()),
                "mse_raw": float(mse_raw.detach().item()),
                "target_mean": self.target_mean,
                "target_std": self.target_std,
                **aux_metrics,
            },
        )

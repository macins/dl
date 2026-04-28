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

        self.lam_cos = float(lam_cos)
        self.lam_mse = float(lam_mse)

        self.pred_key = str(pred_key)
        self.target_key = str(target_key)

        self.target_mean = float(target_mean)
        self.target_std = float(target_std) if float(target_std) != 0.0 else 1.0

        self.pred_index = pred_index
        self.target_index = target_index

    def unnormalize_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        return y_pred * self.target_std + self.target_mean

    def normalize_target(self, y_true: torch.Tensor) -> torch.Tensor:
        return (y_true - self.target_mean) / self.target_std

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

        # If already scalar-per-token, for example:
        #   (B, T)
        #   (B, N, T)
        # return as is.
        if value.ndim < 3:
            return value

        # If last dim is singleton target dim:
        #   (B, T, 1)
        #   (B, N, T, 1)
        # squeeze it.
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
           Used by panel virtual symbol-day minibatch training.
           It controls which labels participate in the current loss.

        2. targets["valid_mask"]
           Optional target-specific mask.

        3. batch["padding_mask"]
           Default full valid input/target mask.

        Important:
            loss_mask is only for objective/metric.
            padding_mask should still be used by model/backbone for attention masking.
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

        self._validate_shapes(y_pred=y_pred, y_true=y_true, mask=mask)

        y_pred_unnorm = self.unnormalize_prediction(y_pred)
        y_true_norm = self.normalize_target(y_true)

        valid = mask.bool()

        if valid.any():
            cos = masked_cosine_similarity(y_pred_unnorm, y_true, valid)
            mse_raw = torch.mean((y_pred_unnorm[valid] - y_true[valid]) ** 2)
            mse_normalized = torch.mean((y_pred[valid] - y_true_norm[valid]) ** 2)
        else:
            cos = torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)
            mse_raw = torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)
            mse_normalized = torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)

        aux_loss_total, aux_metrics = self.get_aux_losses(outputs)

        loss = self.lam_mse * mse_normalized - self.lam_cos * cos + aux_loss_total

        return ObjectiveOutput(
            loss=loss,
            metrics={
                "cosine_similarity": float(cos.detach().item()),
                "mse": float(mse_normalized.detach().item()),
                "mse_raw": float(mse_raw.detach().item()),
                "target_mean": self.target_mean,
                "target_std": self.target_std,
                **aux_metrics,
            },
        )
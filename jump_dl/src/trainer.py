from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import CosineSimilarityMetric
from .utils.externals import ensure_torch

torch = ensure_torch()

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


@dataclass
class EMAConfig:
    enabled: bool = True
    decay: float = 0.99
    start_step: int = 0
    eval_with_ema: bool = True
    save_ema_checkpoint: bool = True


@dataclass
class TrainerConfig:
    num_epochs: int = 10
    device: str = "cuda"
    monitor_key: str = "cosine_similarity"
    monitor_mode: str = "max"
    grad_clip_norm: float | None = None
    log_timing: bool = True
    show_progress_bar: bool = True
    normalize_continuous_features: bool = True
    normalization_eps: float = 0
    print_epoch_table: bool = True
    ema: EMAConfig | dict[str, Any] = field(default_factory=EMAConfig)

    def __post_init__(self) -> None:
        if isinstance(self.ema, dict):
            self.ema = EMAConfig(**self.ema)
        elif not isinstance(self.ema, EMAConfig):
            raise TypeError(f"trainer.ema must be a mapping or EMAConfig, got {type(self.ema)!r}")


class Trainer:
    def __init__(
        self,
        *,
        model,
        objective,
        optimizer,
        scheduler=None,
        feature_stats: dict[str, dict[str, Any]] | None = None,
        config: TrainerConfig | None = None,
        output_dir: str | Path = "workdirs/jump_dl",
    ) -> None:
        self.model = model
        self.objective = objective
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or TrainerConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        self.feature_stats = self._prepare_feature_stats(feature_stats)
        self._epoch_table_header_printed = False
        self.global_step = 0
        self.ema_num_updates = 0
        self.ema_model = None

    def _move_to_device(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._move_to_device(v) for k, v in obj.items()}
        if torch.is_tensor(obj):
            return obj.to(self.device)
        return obj

    def _prepare_feature_stats(self, feature_stats: dict[str, dict[str, Any]] | None) -> dict[str, dict[str, torch.Tensor]]:
        if not feature_stats:
            return {}

        prepared: dict[str, dict[str, torch.Tensor]] = {}
        for group_name, stats in feature_stats.items():
            mean = torch.as_tensor(np.asarray(stats["mean"], dtype=np.float32), device=self.device)
            std = torch.as_tensor(np.asarray(stats["std"], dtype=np.float32), device=self.device)
            std = torch.clamp(std, min=float(self.config.normalization_eps))
            prepared[str(group_name)] = {"mean": mean, "std": std}
        return prepared

    def _normalize_batch_features(self, batch: dict[str, Any]) -> dict[str, Any]:
        if not self.config.normalize_continuous_features or not self.feature_stats:
            return batch

        features = batch.get("features")
        padding_mask = batch.get("padding_mask")
        if not isinstance(features, dict) or not torch.is_tensor(padding_mask):
            return batch

        mask = padding_mask.bool().unsqueeze(-1)
        for group_name, stats in self.feature_stats.items():
            x = features.get(group_name)
            if x is None or not torch.is_tensor(x):
                continue
            mean = stats["mean"].view(1, 1, -1)
            std = stats["std"].view(1, 1, -1)
            x_norm = (x.float() - mean) / std
            features[group_name] = torch.where(mask, x_norm, torch.zeros_like(x_norm))
        return batch

    def _init_ema_model(self) -> None:
        if self.ema_model is not None:
            return
        self.ema_model = copy.deepcopy(self.model).to(self.device)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def _update_ema(self) -> None:
        if not self.config.ema.enabled:
            return
        if self.global_step < self.config.ema.start_step:
            return
        if self.ema_model is None:
            self._init_ema_model()

        decay = float(self.config.ema.decay)
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"trainer.ema.decay must be in [0, 1], got {decay}")

        with torch.no_grad():
            online_state = self.model.state_dict()
            ema_state = self.ema_model.state_dict()
            for name, online_tensor in online_state.items():
                ema_tensor = ema_state[name]
                if torch.is_floating_point(online_tensor):
                    ema_tensor.mul_(decay).add_(online_tensor.detach(), alpha=1.0 - decay)
                else:
                    ema_tensor.copy_(online_tensor)
        self.ema_num_updates += 1

    def _get_eval_model(self):
        if self.config.ema.enabled and self.config.ema.eval_with_ema and self.ema_model is not None:
            return self.ema_model
        return self.model

    def _iter_with_progress(self, dataloader, *, train: bool, epoch: int):
        iterator = dataloader
        if not self.config.show_progress_bar or tqdm is None:
            return iterator

        total = None
        try:
            total = len(dataloader)
        except TypeError:
            total = None

        desc = f"Epoch {epoch}/{self.config.num_epochs} {'train' if train else 'val'}"
        return tqdm(
            dataloader,
            total=total,
            desc=desc,
            leave=False,
            dynamic_ncols=True,
        )

    def _run_epoch(self, dataloader, *, train: bool, epoch: int) -> dict[str, float]:
        metric = CosineSimilarityMetric()
        total_loss = 0.0
        total_steps = 0
        total_data_time = 0.0
        total_forward_time = 0.0
        total_backward_time = 0.0
        total_step_time = 0.0
        metric_sums: dict[str, float] = {}

        run_model = self.model if train else self._get_eval_model()
        run_model.train(train)
        progress = self._iter_with_progress(dataloader, train=train, epoch=epoch)
        iterator = iter(progress)
        while True:
            fetch_start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            total_data_time += time.perf_counter() - fetch_start

            batch = self._move_to_device(batch)
            batch = self._normalize_batch_features(batch)
            if train:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                forward_start = time.perf_counter()
                outputs = run_model(batch)
                step_out = self.objective(outputs, batch)
                loss = step_out.loss
                total_forward_time += time.perf_counter() - forward_start

            if train:
                backward_start = time.perf_counter()
                loss.backward()
                if self.config.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                total_backward_time += time.perf_counter() - backward_start

                step_start = time.perf_counter()
                self.optimizer.step()
                self.global_step += 1
                self._update_ema()
                total_step_time += time.perf_counter() - step_start

            y_pred = self.objective.unnormalize_prediction(self.objective.get_prediction_tensor(outputs)).detach()
            y_true = self.objective.get_target_tensor(batch).detach()
            mask = self.objective.get_mask_tensor(batch).detach()
            metric.update(y_pred, y_true, mask)
            total_loss += float(loss.detach().item())
            total_steps += 1
            for name, value in step_out.metrics.items():
                metric_sums[name] = metric_sums.get(name, 0.0) + float(value)

            if tqdm is not None and self.config.show_progress_bar and hasattr(progress, "set_postfix"):
                progress.set_postfix(
                    loss=f"{total_loss / total_steps:.4f}",
                    cos=f"{metric.compute():.4f}",
                    refresh=False,
                )

        if train and self.scheduler is not None:
            self.scheduler.step()

        out = {
            "loss": total_loss / max(total_steps, 1),
            "cosine_similarity": metric.compute(),
        }
        denom = max(total_steps, 1)
        for name, total in metric_sums.items():
            out[name] = total / denom
        if self.config.log_timing:
            out["avg_data_time"] = total_data_time / denom
            out["avg_forward_time"] = total_forward_time / denom
            out["avg_backward_time"] = total_backward_time / denom if train else 0.0
            out["avg_step_time"] = total_step_time / denom if train else 0.0
        return out

    def _print_epoch_table_header(self, *, has_val: bool) -> None:
        val_prefix = "val_ema" if (self.config.ema.enabled and self.config.ema.eval_with_ema) else "val"
        columns: list[tuple[str, int]] = [
            ("epoch", 10),
            ("train_mse", 12),
            ("train_cos", 12),
        ]
        if has_val:
            columns.extend([
                (f"{val_prefix}_mse", 12),
                (f"{val_prefix}_cos", 12),
            ])
        columns.append(("lr", 12))
        header = " | ".join(f"{name:>{width}}" for name, width in columns)
        divider = "-+-".join("-" * width for _, width in columns)
        print(header, flush=True)
        print(divider, flush=True)
        self._epoch_table_header_printed = True

    def _print_epoch_table_row(
        self,
        *,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> None:
        lr = float(self.optimizer.param_groups[0]["lr"]) if self.optimizer.param_groups else 0.0
        values: list[tuple[str, int]] = [
            (f"{epoch}/{self.config.num_epochs}", 10),
            (f"{train_metrics.get('mse_raw', 0.0):.6f}", 12),
            (f"{train_metrics.get('cosine_similarity', 0.0):.6f}", 12),
        ]
        if val_metrics is not None:
            values.extend([
                (f"{val_metrics.get('mse_raw', 0.0):.6f}", 12),
                (f"{val_metrics.get('cosine_similarity', 0.0):.6f}", 12),
            ])
        values.append((f"{lr:.6e}", 12))
        print(" | ".join(f"{value:>{width}}" for value, width in values), flush=True)

    def fit(self, train_dataloader, val_dataloader=None) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        best_value = None
        self._epoch_table_header_printed = False

        for epoch in range(1, self.config.num_epochs + 1):
            train_metrics = self._run_epoch(train_dataloader, train=True, epoch=epoch)
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_cosine_similarity": train_metrics["cosine_similarity"],
            }
            if "mse" in train_metrics:
                row["train_mse"] = train_metrics["mse"]
            if "mse_raw" in train_metrics:
                row["train_mse_raw"] = train_metrics["mse_raw"]
            if self.config.log_timing:
                row["train_avg_data_time"] = train_metrics.get("avg_data_time", 0.0)
                row["train_avg_forward_time"] = train_metrics.get("avg_forward_time", 0.0)
                row["train_avg_backward_time"] = train_metrics.get("avg_backward_time", 0.0)
                row["train_avg_step_time"] = train_metrics.get("avg_step_time", 0.0)
            if val_dataloader is not None:
                with torch.no_grad():
                    val_metrics = self._run_epoch(val_dataloader, train=False, epoch=epoch)
                row["val_loss"] = val_metrics["loss"]
                row["val_cosine_similarity"] = val_metrics["cosine_similarity"]
                if "mse" in val_metrics:
                    row["val_mse"] = val_metrics["mse"]
                if "mse_raw" in val_metrics:
                    row["val_mse_raw"] = val_metrics["mse_raw"]
                if self.config.log_timing:
                    row["val_avg_data_time"] = val_metrics.get("avg_data_time", 0.0)
                    row["val_avg_forward_time"] = val_metrics.get("avg_forward_time", 0.0)
                current = row[f"val_{self.config.monitor_key}"]
            else:
                current = row[f"train_{self.config.monitor_key}"]

            history.append(row)
            self._save_history(history)
            if self._is_better(current, best_value):
                best_value = current
                self._save_checkpoint("best.pt", epoch=epoch, metrics=row)
            self._save_checkpoint("last.pt", epoch=epoch, metrics=row)
            if self.config.print_epoch_table:
                if not self._epoch_table_header_printed:
                    self._print_epoch_table_header(has_val=val_dataloader is not None)
                self._print_epoch_table_row(epoch=epoch, train_metrics=train_metrics, val_metrics=val_metrics if val_dataloader is not None else None)
            else:
                print(json.dumps(row, ensure_ascii=False), flush=True)
        return history

    def _is_better(self, current: float, best: float | None) -> bool:
        if best is None:
            return True
        if self.config.monitor_mode == "max":
            return current > best
        return current < best

    def _save_history(self, history: list[dict[str, float]]) -> None:
        (self.output_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.output_dir / "trainer_config.json").write_text(
            json.dumps(asdict(self.config), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _save_checkpoint(self, name: str, *, epoch: int, metrics: dict[str, float]) -> None:
        path = self.output_dir / name
        state = {
            "epoch": epoch,
            "metrics": metrics,
            "global_step": self.global_step,
            "ema_num_updates": self.ema_num_updates,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": None if self.scheduler is None else self.scheduler.state_dict(),
        }
        if self.config.ema.enabled and self.config.ema.save_ema_checkpoint and self.ema_model is not None:
            state["ema_state_dict"] = self.ema_model.state_dict()
        torch.save(state, path)

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from torch.nn.parameter import UninitializedBuffer, UninitializedParameter

from .metrics import CosineSimilarityMetric
from .optim.nexus import NexusConfig, NexusEngine
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
class PanelVirtualBatchConfig:
    enabled: bool = False

    # 设成旧版 symbol-day dataloader 的 batch_size。
    # 例如旧版 batch_size=32，这里就设 32。
    virtual_symbol_day_batch_size: int = 32

    # 每个 panel batch 内是否随机打乱 (b, n) pairs。
    shuffle_pairs: bool = True

    # 只在 padding_mask.ndim == 3, 即 (B, N, T) 时启用。
    # 旧版 (B, T) 不会受影响。
    only_for_panel: bool = True


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

    panel_virtual_batch: PanelVirtualBatchConfig | dict[str, Any] = field(
        default_factory=PanelVirtualBatchConfig
    )

    # AMP / autocast
    use_amp: bool = False
    amp_dtype: str = "bfloat16"
    grad_accum_steps: int = 1
    use_nexus: bool = False
    nexus: NexusConfig | dict[str, Any] = field(default_factory=NexusConfig)

    # Printed if present in metrics.
    # Full history.json still stores every metric.
    table_metric_keys: list[str] = field(
        default_factory=lambda: [
            "mse_raw",
            "cosine_similarity",
            "mog_nll",
        ]
    )
    enable_tensorboard: bool = False
    tensorboard_dirname: str = "tensorboard"

    def __post_init__(self) -> None:
        if isinstance(self.ema, dict):
            self.ema = EMAConfig(**self.ema)
        elif not isinstance(self.ema, EMAConfig):
            raise TypeError(f"trainer.ema must be a mapping or EMAConfig, got {type(self.ema)!r}")

        if isinstance(self.panel_virtual_batch, dict):
            self.panel_virtual_batch = PanelVirtualBatchConfig(**self.panel_virtual_batch)
        elif not isinstance(self.panel_virtual_batch, PanelVirtualBatchConfig):
            raise TypeError(
                "trainer.panel_virtual_batch must be a mapping or PanelVirtualBatchConfig, "
                f"got {type(self.panel_virtual_batch)!r}"
            )

        if isinstance(self.nexus, dict):
            self.nexus = NexusConfig(**self.nexus)
        elif not isinstance(self.nexus, NexusConfig):
            raise TypeError(f"trainer.nexus must be a mapping or NexusConfig, got {type(self.nexus)!r}")

        if self.grad_accum_steps < 1:
            raise ValueError("trainer.grad_accum_steps must be >= 1")


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
        self._epoch_table_keys: list[str] = []
        self.global_step = 0
        self.ema_num_updates = 0
        self.nexus_engine: NexusEngine | None = None

        # EMA 不再维护 deepcopy 出来的 ema_model。
        # 只维护 detached cloned state_dict，避免 LazyModule / non-leaf Tensor deepcopy 报错。
        self.ema_state_dict: dict[str, torch.Tensor] | None = None
        self.tb_writer = None
        if self.config.enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except Exception:  # pragma: no cover
                SummaryWriter = None
            if SummaryWriter is not None:
                tb_dir = self.output_dir / str(self.config.tensorboard_dirname)
                tb_dir.mkdir(parents=True, exist_ok=True)
                self.tb_writer = SummaryWriter(log_dir=str(tb_dir))

    @staticmethod
    def _is_uninitialized_tensor(x: Any) -> bool:
        return isinstance(x, (UninitializedParameter, UninitializedBuffer))

    def _set_objective_progress(self, *, train: bool, epoch: int) -> None:
        hook = getattr(self.objective, "set_training_progress", None)
        if callable(hook):
            hook(
                global_step=self.global_step,
                epoch=epoch,
                num_epochs=self.config.num_epochs,
                train=train,
            )

    def _get_amp_dtype(self) -> torch.dtype | None:
        dtype = str(self.config.amp_dtype).strip().lower()

        if dtype in {"bf16", "bfloat16"}:
            return torch.bfloat16

        if dtype in {"fp16", "float16", "half"}:
            return torch.float16

        if dtype in {"fp32", "float32", "none", "off", "false"}:
            return None

        raise ValueError(
            f"Unknown trainer.amp_dtype={self.config.amp_dtype!r}. "
            "Use 'bfloat16', 'float16', or 'float32'."
        )

    def _amp_enabled(self) -> bool:
        return (
            bool(self.config.use_amp)
            and self._get_amp_dtype() is not None
            and self.device.type in {"cuda", "cpu"}
        )

    def _move_to_device(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._move_to_device(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._move_to_device(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._move_to_device(v) for v in obj)
        if torch.is_tensor(obj):
            return obj.to(self.device)
        return obj

    def _prepare_feature_stats(
        self,
        feature_stats: dict[str, dict[str, Any]] | None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        if not feature_stats:
            return {}

        prepared: dict[str, dict[str, torch.Tensor]] = {}

        eps = float(self.config.normalization_eps)
        if eps <= 0.0:
            eps = 1e-12

        for group_name, stats in feature_stats.items():
            mean = torch.as_tensor(
                np.asarray(stats["mean"], dtype=np.float32),
                device=self.device,
            )
            std = torch.as_tensor(
                np.asarray(stats["std"], dtype=np.float32),
                device=self.device,
            )

            std = torch.where(
                torch.isfinite(std) & (std.abs() >= eps),
                std,
                torch.ones_like(std),
            )
            mean = torch.where(
                torch.isfinite(mean),
                mean,
                torch.zeros_like(mean),
            )

            prepared[str(group_name)] = {"mean": mean, "std": std}

        return prepared

    def _normalize_batch_features(self, batch: dict[str, Any]) -> dict[str, Any]:
        if not self.config.normalize_continuous_features or not self.feature_stats:
            return batch

        features = batch.get("features")
        padding_mask = batch.get("padding_mask")

        if not isinstance(features, dict) or not torch.is_tensor(padding_mask):
            return batch

        # old:
        #   x:            (B, T, F)
        #   padding_mask: (B, T)
        #
        # panel:
        #   x:            (B, N, T, F)
        #   padding_mask: (B, N, T)
        mask = padding_mask.bool().unsqueeze(-1)

        for group_name, stats in self.feature_stats.items():
            x = features.get(group_name)
            if x is None or not torch.is_tensor(x):
                continue

            mean = stats["mean"]
            std = stats["std"]

            if x.shape[-1] != mean.numel():
                raise ValueError(
                    f"Feature stats dim mismatch for group={group_name!r}: "
                    f"x.shape[-1]={x.shape[-1]}, stats_dim={mean.numel()}."
                )

            view_shape = (1,) * (x.ndim - 1) + (-1,)
            mean = mean.view(*view_shape)
            std = std.view(*view_shape)

            x_norm = (x.float() - mean) / std
            features[group_name] = torch.where(mask, x_norm, torch.zeros_like(x_norm))

        return batch

    # ---------------------------------------------------------------------
    # Panel virtual symbol-day minibatch
    # ---------------------------------------------------------------------

    def _make_virtual_symbol_day_loss_masks(
        self,
        padding_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        padding_mask = padding_mask.bool()
        cfg = self.config.panel_virtual_batch

        if not cfg.enabled:
            return [padding_mask]

        if cfg.only_for_panel and padding_mask.ndim == 2:
            return [padding_mask]

        if padding_mask.ndim == 2:
            return [padding_mask]

        if padding_mask.ndim != 3:
            raise ValueError(
                f"panel_virtual_batch expects padding_mask with shape (B,T) or (B,N,T), "
                f"got {tuple(padding_mask.shape)}."
            )

        B, N, _T = padding_mask.shape

        virtual_batch_size = int(cfg.virtual_symbol_day_batch_size)
        if virtual_batch_size <= 0:
            raise ValueError(
                "trainer.panel_virtual_batch.virtual_symbol_day_batch_size "
                f"must be positive, got {virtual_batch_size}."
            )

        pair_valid = padding_mask.any(dim=-1)  # (B, N)
        pair_indices = pair_valid.nonzero(as_tuple=False)  # (M, 2), columns: b, n

        if pair_indices.numel() == 0:
            return [padding_mask]

        if cfg.shuffle_pairs:
            perm = torch.randperm(pair_indices.shape[0], device=pair_indices.device)
            pair_indices = pair_indices[perm]

        masks: list[torch.Tensor] = []

        for start in range(0, pair_indices.shape[0], virtual_batch_size):
            chunk = pair_indices[start:start + virtual_batch_size]

            pair_mask = torch.zeros(
                (B, N),
                dtype=torch.bool,
                device=padding_mask.device,
            )
            pair_mask[chunk[:, 0], chunk[:, 1]] = True

            loss_mask = padding_mask & pair_mask.unsqueeze(-1)
            masks.append(loss_mask)

        return masks

    # ---------------------------------------------------------------------
    # EMA as state_dict, no deepcopy(model)
    # ---------------------------------------------------------------------

    def _init_ema_state(self) -> None:
        ema_state: dict[str, torch.Tensor] = {}

        for name, tensor in self.model.state_dict().items():
            if self._is_uninitialized_tensor(tensor):
                continue
            ema_state[name] = tensor.detach().clone()

        self.ema_state_dict = ema_state

    def _update_ema(self) -> None:
        if not self.config.ema.enabled:
            return

        if self.global_step < self.config.ema.start_step:
            return

        if self.ema_state_dict is None:
            self._init_ema_state()
            self.ema_num_updates += 1
            return

        decay = float(self.config.ema.decay)
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"trainer.ema.decay must be in [0, 1], got {decay}")

        online_state = self.model.state_dict()

        with torch.no_grad():
            for name, online_tensor in online_state.items():
                if self._is_uninitialized_tensor(online_tensor):
                    continue

                online_tensor = online_tensor.detach()
                ema_tensor = self.ema_state_dict.get(name)

                if (
                    ema_tensor is None
                    or tuple(ema_tensor.shape) != tuple(online_tensor.shape)
                    or ema_tensor.dtype != online_tensor.dtype
                    or ema_tensor.device != online_tensor.device
                ):
                    self.ema_state_dict[name] = online_tensor.clone()
                    continue

                if torch.is_floating_point(online_tensor):
                    ema_tensor.mul_(decay).add_(online_tensor, alpha=1.0 - decay)
                else:
                    ema_tensor.copy_(online_tensor)

        self.ema_num_updates += 1

    def _swap_in_ema_weights(self):
        if (
            not self.config.ema.enabled
            or not self.config.ema.eval_with_ema
            or self.ema_state_dict is None
        ):
            return None

        model_state = self.model.state_dict()
        backup: dict[str, torch.Tensor] = {}

        with torch.no_grad():
            for name, ema_tensor in self.ema_state_dict.items():
                model_tensor = model_state.get(name)

                if model_tensor is None:
                    continue
                if self._is_uninitialized_tensor(model_tensor):
                    continue
                if tuple(model_tensor.shape) != tuple(ema_tensor.shape):
                    continue

                backup[name] = model_tensor.detach().clone()
                model_tensor.copy_(
                    ema_tensor.to(
                        device=model_tensor.device,
                        dtype=model_tensor.dtype,
                    )
                )

        def restore() -> None:
            current_state = self.model.state_dict()
            with torch.no_grad():
                for name, old_tensor in backup.items():
                    model_tensor = current_state.get(name)
                    if model_tensor is None:
                        continue
                    if self._is_uninitialized_tensor(model_tensor):
                        continue
                    model_tensor.copy_(
                        old_tensor.to(
                            device=model_tensor.device,
                            dtype=model_tensor.dtype,
                        )
                    )

        return restore

    # ---------------------------------------------------------------------
    # Epoch loop
    # ---------------------------------------------------------------------

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

    def _run_one_step(
        self,
        *,
        batch: dict[str, Any],
        run_model,
        train: bool,
        epoch: int,
    ) -> tuple[dict[str, Any], Any, float, float, float]:
        """
        Run one forward/backward/optimizer step.

        Returns:
            outputs
            step_out
            forward_time
            backward_time
            step_time
        """
        if train:
            self.optimizer.zero_grad(set_to_none=True)

        self._set_objective_progress(train=train, epoch=epoch)

        amp_dtype = self._get_amp_dtype()
        amp_enabled = self._amp_enabled()

        with torch.set_grad_enabled(train):
            forward_start = time.perf_counter()

            with torch.autocast(
                device_type=self.device.type,
                dtype=amp_dtype if amp_dtype is not None else torch.float32,
                enabled=amp_enabled,
            ):
                outputs = run_model(batch)
                step_out = self.objective(outputs, batch)
                loss = step_out.loss

            forward_time = time.perf_counter() - forward_start

        backward_time = 0.0
        step_time = 0.0

        if train:
            backward_start = time.perf_counter()
            loss.backward()

            if self.config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip_norm,
                )

            backward_time = time.perf_counter() - backward_start

            step_start = time.perf_counter()
            self.optimizer.step()
            self.global_step += 1
            self._update_ema()
            step_time = time.perf_counter() - step_start

        return outputs, step_out, forward_time, backward_time, step_time

    def _run_epoch(self, dataloader, *, train: bool, epoch: int) -> dict[str, float]:
        metric = CosineSimilarityMetric()

        total_loss = 0.0
        total_steps = 0

        total_data_time = 0.0
        total_forward_time = 0.0
        total_backward_time = 0.0
        total_step_time = 0.0

        metric_sums: dict[str, float] = {}

        ema_restore = None
        if not train:
            ema_restore = self._swap_in_ema_weights()

        run_model = self.model
        run_model.train(train)
        nexus_enabled = bool(train and (self.config.use_nexus or self.config.nexus.enabled))
        if nexus_enabled:
            if hasattr(self.model, "module"):
                raise NotImplementedError("Nexus with DDP is not implemented yet")
            self.nexus_engine = NexusEngine(
                model=self.model,
                inner_lr=self.config.nexus.inner_lr,
                eps=self.config.nexus.eps,
                normalize_scope=self.config.nexus.normalize_scope,
                copy_buffers=self.config.nexus.copy_buffers,
            )
            self.nexus_engine.sync_inner_from_main()

        try:
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
                    padding_mask = batch.get("padding_mask")
                    if not torch.is_tensor(padding_mask):
                        raise KeyError("Batch must contain tensor 'padding_mask'.")

                    loss_masks = self._make_virtual_symbol_day_loss_masks(padding_mask)
                else:
                    # Validation: full mask, no stochastic loss_mask.
                    loss_masks = [None]

                for loss_mask in loss_masks:
                    if loss_mask is None:
                        step_batch = batch
                    else:
                        step_batch = dict(batch)
                        step_batch["loss_mask"] = loss_mask

                    if not nexus_enabled:
                        outputs, step_out, fwd_t, bwd_t, step_t = self._run_one_step(
                            batch=step_batch,
                            run_model=run_model,
                            train=train,
                            epoch=epoch,
                        )
                    else:
                        self._set_objective_progress(train=train, epoch=epoch)
                        amp_dtype = self._get_amp_dtype()
                        amp_enabled = self._amp_enabled()
                        forward_start = time.perf_counter()
                        with torch.autocast(
                            device_type=self.device.type,
                            dtype=amp_dtype if amp_dtype is not None else torch.float32,
                            enabled=amp_enabled,
                        ):
                            outputs = self.nexus_engine.inner_model(step_batch)
                            step_out = self.objective(outputs, step_batch)
                            loss = step_out.loss
                        fwd_t = time.perf_counter() - forward_start
                        backward_start = time.perf_counter()
                        loss.backward()
                        self.nexus_engine.inner_step()
                        self.nexus_engine.zero_inner_grad()
                        bwd_t = time.perf_counter() - backward_start
                        step_t = 0.0

                    total_forward_time += fwd_t
                    total_backward_time += bwd_t
                    total_step_time += step_t

                    y_pred = self.objective.unnormalize_prediction(
                        self.objective.get_prediction_tensor(outputs)
                    ).detach()
                    y_true = self.objective.get_target_tensor(step_batch).detach()
                    mask = self.objective.get_mask_tensor(step_batch).detach()

                    metric.update(y_pred, y_true, mask)

                    total_loss += float(step_out.loss.detach().item())
                    total_steps += 1

                    if nexus_enabled and self.config.nexus.log_inner_loss:
                        metric_sums["nexus_inner_loss"] = metric_sums.get("nexus_inner_loss", 0.0) + float(
                            step_out.loss.detach().item()
                        )

                    if nexus_enabled and (total_steps % self.config.grad_accum_steps == 0):
                        step_start = time.perf_counter()
                        self.optimizer.zero_grad(set_to_none=True)
                        pseudo_stats = self.nexus_engine.assign_pseudo_grad_to_main()
                        if self.config.nexus.log_pseudo_grad_norm:
                            metric_sums["nexus_pseudo_grad_norm"] = metric_sums.get("nexus_pseudo_grad_norm", 0.0) + float(
                                pseudo_stats["pseudo_grad_norm"]
                            )
                            metric_sums["nexus_inner_delta_norm"] = metric_sums.get("nexus_inner_delta_norm", 0.0) + float(
                                pseudo_stats["inner_delta_norm"]
                            )
                        metric_sums["nexus_inner_lr"] = metric_sums.get("nexus_inner_lr", 0.0) + float(self.config.nexus.inner_lr)
                        if self.config.grad_clip_norm is not None:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.global_step += 1
                        self._update_ema()
                        self.nexus_engine.sync_inner_from_main()
                        total_step_time += time.perf_counter() - step_start

                    for name, value in step_out.metrics.items():
                        metric_sums[name] = metric_sums.get(name, 0.0) + float(value)

                    if (
                        tqdm is not None
                        and self.config.show_progress_bar
                        and hasattr(progress, "set_postfix")
                    ):
                        postfix = {
                            "loss": f"{total_loss / max(total_steps, 1):.4f}",
                            "cos": f"{metric.compute():.4f}",
                            "steps": str(total_steps),
                            "amp": str(self.config.use_amp),
                            "dtype": str(self.config.amp_dtype),
                        }

                        if "mog_nll" in metric_sums:
                            postfix["mog"] = f"{metric_sums['mog_nll'] / max(total_steps, 1):.4f}"

                        progress.set_postfix(**postfix, refresh=False)

            if nexus_enabled and (total_steps % self.config.grad_accum_steps != 0):
                step_start = time.perf_counter()
                self.optimizer.zero_grad(set_to_none=True)
                pseudo_stats = self.nexus_engine.assign_pseudo_grad_to_main()
                if self.config.nexus.log_pseudo_grad_norm:
                    metric_sums["nexus_pseudo_grad_norm"] = metric_sums.get("nexus_pseudo_grad_norm", 0.0) + float(
                        pseudo_stats["pseudo_grad_norm"]
                    )
                    metric_sums["nexus_inner_delta_norm"] = metric_sums.get("nexus_inner_delta_norm", 0.0) + float(
                        pseudo_stats["inner_delta_norm"]
                    )
                metric_sums["nexus_inner_lr"] = metric_sums.get("nexus_inner_lr", 0.0) + float(self.config.nexus.inner_lr)
                if self.config.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                self._update_ema()
                self.nexus_engine.sync_inner_from_main()
                total_step_time += time.perf_counter() - step_start

            if train and self.scheduler is not None:
                self.scheduler.step()

            denom = max(total_steps, 1)
            
            global_cos = metric.compute()
            
            out = {
                "loss": total_loss / denom,
                "cosine_similarity": global_cos,
            }
            
            for name, total in metric_sums.items():
                avg_value = total / denom
            
                # IMPORTANT:
                # step_out.metrics["cosine_similarity"] is per-step/per-virtual-batch cosine.
                # Do NOT let it overwrite the epoch/global cosine.
                if name == "cosine_similarity":
                    out["step_cosine_similarity"] = avg_value
                else:
                    out[name] = avg_value

            if self.config.log_timing:
                out["avg_data_time"] = total_data_time / denom
                out["avg_forward_time"] = total_forward_time / denom
                out["avg_backward_time"] = total_backward_time / denom if train else 0.0
                out["avg_step_time"] = total_step_time / denom if train else 0.0

            return out

        finally:
            if ema_restore is not None:
                ema_restore()

    # ---------------------------------------------------------------------
    # Logging / fit / checkpoint
    # ---------------------------------------------------------------------
    @staticmethod
    def _format_metric_name(name: str) -> str:
        mapping = {
            "loss": "loss",
            "cosine_similarity": "cos",
            "mse_raw": "mse",
            "mse": "mseN",
            "mog_nll": "mog",
            "loss_weight_cos": "wC",
            "loss_weight_mse": "wM",
            "loss_weight_mog_nll": "wG",
        }
        return mapping.get(name, name)
    
    
    def _print_epoch_table_header(
        self,
        *,
        has_val: bool,
        metric_keys: list[str],
    ) -> None:
        columns: list[tuple[str, int]] = [("ep", 7)]
    
        for key in metric_keys:
            columns.append((f"tr_{self._format_metric_name(key)}", 9))
    
        if has_val:
            val_prefix = "ve" if (
                self.config.ema.enabled and self.config.ema.eval_with_ema
            ) else "va"
    
            for key in metric_keys:
                columns.append((f"{val_prefix}_{self._format_metric_name(key)}", 9))
    
        columns.append(("lr", 10))
    
        header = " ".join(f"{name:>{width}}" for name, width in columns)
        divider = " ".join("-" * width for _, width in columns)
    
        print(header, flush=True)
        print(divider, flush=True)
    
        self._epoch_table_header_printed = True
        self._epoch_table_keys = metric_keys
    
    
    def _print_epoch_table_row(
        self,
        *,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> None:
        lr = float(self.optimizer.param_groups[0]["lr"]) if self.optimizer.param_groups else 0.0
    
        values: list[tuple[str, int]] = [
            (f"{epoch}/{self.config.num_epochs}", 7),
        ]
    
        def fmt(key: str, value: float | None) -> str:
            if value is None:
                return ""
    
            if key in {"cosine_similarity"}:
                return f"{value:.4f}"
    
            if key in {"mse_raw", "mse"}:
                return f"{value:.2e}"
    
            if key in {"mog_nll"}:
                return f"{value:.3f}"
    
            if key.startswith("loss_weight_"):
                return f"{value:.2f}"
    
            return f"{value:.4f}"
    
        for key in self._epoch_table_keys:
            values.append((fmt(key, train_metrics.get(key)), 9))
    
        if val_metrics is not None:
            for key in self._epoch_table_keys:
                values.append((fmt(key, val_metrics.get(key)), 9))
    
        values.append((f"{lr:.2e}", 10))
    
        print(" ".join(f"{value:>{width}}" for value, width in values), flush=True)

    def _available_table_keys(
        self,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> list[str]:
        keys: list[str] = []

        for key in self.config.table_metric_keys:
            if key in train_metrics or (val_metrics is not None and key in val_metrics):
                keys.append(key)

        if not keys:
            keys = ["loss"]

        return keys

    def fit(self, train_dataloader, val_dataloader=None) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        best_value = None
        self._epoch_table_header_printed = False
        self._epoch_table_keys = []

        for epoch in range(1, self.config.num_epochs + 1):
            train_metrics = self._run_epoch(
                train_dataloader,
                train=True,
                epoch=epoch,
            )

            row: dict[str, float | int] = {"epoch": epoch}

            for name, value in train_metrics.items():
                row[f"train_{name}"] = float(value)

            val_metrics = None
            if val_dataloader is not None:
                with torch.no_grad():
                    val_metrics = self._run_epoch(
                        val_dataloader,
                        train=False,
                        epoch=epoch,
                    )

                for name, value in val_metrics.items():
                    row[f"val_{name}"] = float(value)

                monitor_name = f"val_{self.config.monitor_key}"
            else:
                monitor_name = f"train_{self.config.monitor_key}"

            if monitor_name not in row:
                raise KeyError(
                    f"Monitor key {monitor_name!r} not found in logged metrics. "
                    f"Available keys: {sorted(row.keys())}"
                )

            current = float(row[monitor_name])
            history.append(row)  # type: ignore[arg-type]
            self._log_tensorboard_epoch(row)

            self._save_history(history)  # type: ignore[arg-type]

            if self._is_better(current, best_value):
                best_value = current
                self._save_checkpoint("best.pt", epoch=epoch, metrics=row)  # type: ignore[arg-type]

            self._save_checkpoint("last.pt", epoch=epoch, metrics=row)  # type: ignore[arg-type]

            if self.config.print_epoch_table:
                if not self._epoch_table_header_printed:
                    metric_keys = self._available_table_keys(train_metrics, val_metrics)
                    self._print_epoch_table_header(
                        has_val=val_dataloader is not None,
                        metric_keys=metric_keys,
                    )

                self._print_epoch_table_row(
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                )
            else:
                print(json.dumps(row, ensure_ascii=False), flush=True)

        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()

        return history  # type: ignore[return-value]

    def _log_tensorboard_epoch(self, row: dict[str, float | int]) -> None:
        if self.tb_writer is None:
            return

        epoch = int(row.get("epoch", 0))
        for key, value in row.items():
            if key == "epoch":
                continue
            if isinstance(value, (int, float)):
                self.tb_writer.add_scalar(self._tb_tag_for_key(str(key)), float(value), epoch)

        if self.optimizer.param_groups:
            self.tb_writer.add_scalar("optim/lr", float(self.optimizer.param_groups[0]["lr"]), epoch)

    @staticmethod
    def _tb_tag_for_key(key: str) -> str:
        scope = "epoch"
        name = key
        if key.startswith("train_"):
            scope = "train"
            name = key[len("train_"):]
        elif key.startswith("val_"):
            scope = "val"
            name = key[len("val_"):]

        metric_names = {
            "cosine_similarity",
            "step_cosine_similarity",
            "mse",
            "mse_raw",
            "mog_nll",
        }
        if name in metric_names:
            group = "metrics"
        elif name.startswith("loss_weight_"):
            group = "weights"
        elif "loss" in name:
            group = "losses"
        else:
            group = "others"

        return f"{scope}/{group}/{name}"
        
        if self.optimizer.param_groups:
            self.tb_writer.add_scalar("lr", float(self.optimizer.param_groups[0]["lr"]), epoch)

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

    def _save_checkpoint(
        self,
        name: str,
        *,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
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

        if (
            self.config.ema.enabled
            and self.config.ema.save_ema_checkpoint
            and self.ema_state_dict is not None
        ):
            state["ema_state_dict"] = {
                key: value.detach().cpu().clone()
                for key, value in self.ema_state_dict.items()
                if not self._is_uninitialized_tensor(value)
            }

        torch.save(state, path)

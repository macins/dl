from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ..utils.externals import ensure_torch

torch = ensure_torch()
nn = torch.nn
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter


@dataclass
class NexusConfig:
    enabled: bool = False
    inner_lr: float = 1.0e-3
    eps: float = 1.0e-12
    normalize_scope: str = "global"
    copy_buffers: bool = True
    log_inner_loss: bool = True
    log_pseudo_grad_norm: bool = True


class NormalizedSGD:
    def __init__(self, params, lr: float, eps: float = 1e-12, normalize_scope: str = "global") -> None:
        self.params = list(params)
        self.lr = float(lr)
        self.eps = float(eps)
        self.normalize_scope = str(normalize_scope)
        if self.normalize_scope not in {"global", "per_param"}:
            raise ValueError("normalize_scope must be 'global' or 'per_param'")

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            if p.grad is None:
                continue
            if set_to_none:
                p.grad = None
            else:
                p.grad.zero_()

    def step(self) -> None:
        with torch.no_grad():
            if self.normalize_scope == "global":
                total = torch.zeros((), device=self.params[0].device if self.params else "cpu", dtype=torch.float32)
                for p in self.params:
                    if p.grad is None:
                        continue
                    g = p.grad.detach().float()
                    total = total + torch.sum(g * g)
                denom = torch.sqrt(total) + self.eps
                for p in self.params:
                    if p.grad is None:
                        continue
                    upd = p.grad.detach().to(dtype=torch.float32) / denom
                    p.add_(upd.to(dtype=p.dtype), alpha=-self.lr)
            else:
                for p in self.params:
                    if p.grad is None:
                        continue
                    g = p.grad.detach().float()
                    denom = torch.linalg.vector_norm(g) + self.eps
                    upd = g / denom
                    p.add_(upd.to(dtype=p.dtype), alpha=-self.lr)


class NexusEngine:
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float,
        eps: float = 1e-12,
        normalize_scope: str = "global",
        copy_buffers: bool = True,
    ) -> None:
        self.model = model
        self.copy_buffers = bool(copy_buffers)
        self.inner_model = deepcopy(model)
        self.inner_model.to(next(model.parameters()).device)
        self._validate_alignment()
        self.inner_optimizer = NormalizedSGD(
            self.inner_model.parameters(), lr=inner_lr, eps=eps, normalize_scope=normalize_scope
        )

    def _validate_alignment(self) -> None:
        def _is_uninitialized(x: Any) -> bool:
            return isinstance(x, (UninitializedParameter, UninitializedBuffer))

        main_params = list(self.model.named_parameters())
        inner_params = list(self.inner_model.named_parameters())
        if len(main_params) != len(inner_params):
            raise ValueError("Main/inner parameter count mismatch")
        for (mn, mp), (inn, ip) in zip(main_params, inner_params):
            if mn != inn:
                raise ValueError(f"Main/inner parameter name mismatch: {mn} vs {inn}")
            if _is_uninitialized(mp) or _is_uninitialized(ip):
                continue
            if tuple(mp.shape) != tuple(ip.shape):
                raise ValueError(f"Main/inner parameter shape mismatch on {mn}")
            ip.requires_grad_(mp.requires_grad)

    def sync_inner_from_main(self) -> None:
        if self.copy_buffers:
            self.inner_model.load_state_dict(self.model.state_dict(), strict=True)
        else:
            inner_sd = self.inner_model.state_dict()
            for name, tensor in self.model.named_parameters():
                inner_sd[name].copy_(tensor.detach())
        self.inner_model.train(self.model.training)

    def zero_inner_grad(self) -> None:
        self.inner_optimizer.zero_grad(set_to_none=True)

    def inner_step(self) -> None:
        self.inner_optimizer.step()

    def assign_pseudo_grad_to_main(self) -> dict[str, float]:
        pseudo_sq = 0.0
        main_sq = 0.0
        for (mn, mp), (inn, ip) in zip(self.model.named_parameters(), self.inner_model.named_parameters()):
            if mn != inn:
                raise ValueError(f"Main/inner parameter name mismatch: {mn} vs {inn}")
            if not mp.requires_grad:
                continue
            grad = mp.detach() - ip.detach().to(device=mp.device, dtype=mp.dtype)
            mp.grad = grad.clone()
            pseudo_sq += float(torch.sum(grad.detach().float() ** 2).item())
            main_sq += float(torch.sum(mp.detach().float() ** 2).item())
        pseudo_norm = pseudo_sq ** 0.5
        main_norm = main_sq ** 0.5
        ratio = pseudo_norm / (main_norm + 1e-12)
        return {
            "pseudo_grad_norm": pseudo_norm,
            "inner_delta_norm": pseudo_norm,
            "param_delta_ratio": ratio,
        }

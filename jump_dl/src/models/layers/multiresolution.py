from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ...utils.externals import ensure_torch
from .norms import build_norm

torch = ensure_torch()
nn = torch.nn
F = nn.functional


class CausalConv1dTime(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, conv_type: str = "depthwise") -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.kernel_size = int(kernel_size)
        conv_type = str(conv_type).strip().lower()
        if conv_type not in {"depthwise", "regular"}:
            raise ValueError("conv_type must be 'depthwise' or 'regular'.")
        groups = self.d_model if conv_type == "depthwise" else 1
        self.conv = nn.Conv1d(self.d_model, self.d_model, self.kernel_size, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim not in (3, 4):
            raise ValueError(f"Expected x in (B,T,D) or (B,N,T,D), got {tuple(x.shape)}")
        x_flat = x if x.ndim == 3 else x.reshape(-1, x.shape[-2], x.shape[-1])
        xt = x_flat.transpose(1, 2)
        left_pad = self.kernel_size - 1
        yt = self.conv(F.pad(xt, (left_pad, 0)))
        y = yt.transpose(1, 2)
        return y if x.ndim == 3 else y.reshape(*x.shape)


class MultiScaleCausalConv(nn.Module):
    def __init__(self, d_model: int, scales: Sequence[int] = (5, 15, 30), dropout: float = 0.0,
                 conv_type: str = "depthwise", fusion: str = "softmax_gate", use_norm: bool = True,
                 residual_scale: float = 1.0) -> None:
        super().__init__()
        self.scales = [int(s) for s in scales]
        self.fusion = str(fusion).strip().lower()
        self.residual_scale = float(residual_scale)
        self.convs = nn.ModuleList([CausalConv1dTime(d_model, s, conv_type=conv_type) for s in self.scales])
        self.dropout = nn.Dropout(float(dropout))
        self.norm = build_norm("rmsnorm", d_model) if use_norm else nn.Identity()
        self.latest_gate_mean: dict[str, float] = {}

        if self.fusion == "concat":
            self.concat_proj = nn.Linear(d_model * len(self.scales), d_model)
        elif self.fusion == "sigmoid_gate":
            self.sigmoid_gates = nn.ModuleList([nn.Linear(d_model, 1) for _ in self.scales])
        elif self.fusion == "softmax_gate":
            self.softmax_gate = nn.Linear(d_model, len(self.scales))
        elif self.fusion != "sum":
            raise ValueError("fusion must be sum|concat|sigmoid_gate|softmax_gate")

    def _reset_gate_stats(self) -> None:
        self.latest_gate_mean = {}

    def get_aux_stats(self) -> dict[str, float]:
        return dict(self.latest_gate_mean)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._reset_gate_stats()
        base = self.norm(x)
        zs = [conv(base) for conv in self.convs]

        if self.fusion == "sum":
            out = torch.stack(zs, dim=0).sum(dim=0)
        elif self.fusion == "concat":
            out = self.concat_proj(torch.cat(zs, dim=-1))
        elif self.fusion == "sigmoid_gate":
            gates = [torch.sigmoid(g(base)) for g in self.sigmoid_gates]
            out = sum(g * z for g, z in zip(gates, zs))
            for s, g in zip(self.scales, gates):
                self.latest_gate_mean[f"multires/gate_scale_{s}"] = float(g.detach().mean().item())
        else:
            logits = self.softmax_gate(base)
            gate = torch.softmax(logits, dim=-1)
            out = sum(gate[..., i:i+1] * z for i, z in enumerate(zs))
            for i, s in enumerate(self.scales):
                self.latest_gate_mean[f"multires/gate_scale_{s}"] = float(gate[..., i].detach().mean().item())

        return x + self.residual_scale * self.dropout(out)


class MultiResolutionStem(nn.Module):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.multires = MultiScaleCausalConv(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.multires(x)

    def get_aux_stats(self) -> dict[str, float]:
        return self.multires.get_aux_stats()


class MultiResolutionSublayer(nn.Module):
    def __init__(self, d_model: int, norm_type: str = "rmsnorm", norm_eps: float = 1e-6, **kwargs: Any) -> None:
        super().__init__()
        self.norm = build_norm(norm_type, d_model, eps=norm_eps)
        self.multires = MultiScaleCausalConv(d_model=d_model, use_norm=False, **kwargs)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        del padding_mask
        return self.multires(self.norm(x))

    def get_aux_stats(self) -> dict[str, float]:
        return self.multires.get_aux_stats()


class CausalPatchMemoryCrossAttention(nn.Module):
    def __init__(self, d_model: int, scales: Sequence[int] = (5, 15, 30), num_heads: int = 4,
                 dropout: float = 0.0, max_patches: int | None = None, include_partial_patch: bool = False,
                 residual_scale: float = 1.0) -> None:
        super().__init__()
        self.scales = [int(s) for s in scales]
        self.max_patches = max_patches
        self.include_partial_patch = bool(include_partial_patch)
        self.residual_scale = float(residual_scale)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.out_dropout = nn.Dropout(dropout)

    def _build_completed_patch_memory(self, x: torch.Tensor, s: int) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        p = T // s
        if p == 0:
            main = x[:, :0, :]
            ends = torch.zeros((0,), device=x.device, dtype=torch.long)
        else:
            main = x[:, : p * s, :].reshape(B, p, s, D).mean(dim=2)
            ends = torch.arange(s - 1, p * s, s, device=x.device)

        # Always prepend one sentinel memory token that is valid for all query
        # time steps to avoid all-masked attention rows (which can produce NaNs).
        sentinel = torch.zeros(B, 1, D, device=x.device, dtype=x.dtype)
        mem = torch.cat([sentinel, main], dim=1)
        sentinel_end = torch.full((1,), -1, device=x.device, dtype=torch.long)
        ends = torch.cat([sentinel_end, ends], dim=0)

        if self.max_patches is not None and p > self.max_patches:
            keep = int(self.max_patches)
            mem = torch.cat([mem[:, :1, :], mem[:, -keep:, :]], dim=1)
            ends = torch.cat([ends[:1], ends[-keep:]], dim=0)
        return mem, ends

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x3 = x if x.ndim == 3 else x.reshape(-1, x.shape[-2], x.shape[-1])
        B, T, D = x3.shape
        out = torch.zeros_like(x3)
        query = x3
        for s in self.scales:
            mem, ends = self._build_completed_patch_memory(x3, s)
            if mem.shape[1] == 0:
                continue
            allow = ends.view(1, 1, -1) <= torch.arange(T, device=x3.device).view(1, T, 1)
            attn_mask = ~allow.expand(B, T, mem.shape[1]).reshape(B * T, mem.shape[1])
            q = query.reshape(B * T, 1, D)
            k = mem.repeat_interleave(T, dim=0)
            v = k
            attn_out, _ = self.attn(q, k, v, key_padding_mask=attn_mask)
            out = out + attn_out.reshape(B, T, D)
        y = x3 + self.residual_scale * self.out_dropout(out)
        return y if x.ndim == 3 else y.reshape(*x.shape)


class RouterConditionedMultiScale(MultiScaleCausalConv):
    def __init__(self, d_model: int, scales: Sequence[int] = (5, 15, 30), **kwargs: Any) -> None:
        super().__init__(d_model=d_model, scales=scales, fusion="softmax_gate", **kwargs)

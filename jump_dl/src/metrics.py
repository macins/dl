from __future__ import annotations

from dataclasses import dataclass

from .utils.externals import ensure_torch

torch = ensure_torch()


def masked_cosine_similarity(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 0,
) -> torch.Tensor:
    mask = mask.to(dtype=torch.bool)
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    if y_true.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=y_true.device)
    mean_xy = torch.mean(y_true * y_pred)
    mean_y2 = torch.mean(y_true.square())
    mean_pred2 = torch.mean(y_pred.square())
    denom = torch.sqrt(torch.clamp(mean_y2 * mean_pred2, min=eps))
    return mean_xy / denom


@dataclass
class CosineSimilarityMetric:
    eps: float = 0

    def reset(self) -> None:
        self.sum_xy = 0.0
        self.sum_y2 = 0.0
        self.sum_pred2 = 0.0
        self.count = 0

    def __post_init__(self) -> None:
        self.reset()

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> None:
        mask = mask.bool()
        yp = y_pred[mask]
        yt = y_true[mask]
        if yt.numel() == 0:
            return
        self.sum_xy += float((yt * yp).sum().item())
        self.sum_y2 += float(yt.square().sum().item())
        self.sum_pred2 += float(yp.square().sum().item())
        self.count += int(yt.numel())

    def compute(self) -> float:
        if self.count == 0:
            return 0.0
        mean_xy = self.sum_xy / self.count
        mean_y2 = self.sum_y2 / self.count
        mean_pred2 = self.sum_pred2 / self.count
        denom = (max(mean_y2 * mean_pred2, self.eps)) ** 0.5
        return float(mean_xy / denom)


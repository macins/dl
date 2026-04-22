from __future__ import annotations

from ..utils.externals import ensure_torch

torch = ensure_torch()
nn = torch.nn


class BaseModel(nn.Module):
    pass


class BaseEncoder(nn.Module):
    pass


class BaseBackbone(nn.Module):
    output_dim: int


class BaseHead(nn.Module):
    pass

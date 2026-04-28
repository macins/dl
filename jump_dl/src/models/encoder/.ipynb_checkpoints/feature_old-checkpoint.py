from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..base import BaseEncoder
from .registry import register_encoder
from ...utils.externals import ensure_torch

torch = ensure_torch()
nn = torch.nn


@register_encoder("tabular_sequence")
class TabularSequenceEncoder(BaseEncoder):
    def __init__(
        self,
        *,
        numeric_feature_groups: Sequence[str] = ("continuous",),
        categorical_group_name: str = "categorical",
        categorical_cols: Sequence[str] = (),
        vocab_sizes: Mapping[str, int] | None = None,
        categorical_embedding_dim: int = 8,
        model_dim: int = 128,
    ) -> None:
        super().__init__()
        self.numeric_feature_groups = [str(v) for v in numeric_feature_groups]
        self.categorical_group_name = str(categorical_group_name)
        self.categorical_cols = [str(v) for v in categorical_cols]
        self.output_dim = int(model_dim)

        self.category_embeddings = nn.ModuleDict()
        for col in self.categorical_cols:
            vocab_size = {"Symbol": 52, "DATA_sector": 8, "DATA_month": 14}.get(col)
            self.category_embeddings[col] = nn.Embedding(vocab_size, categorical_embedding_dim, padding_idx=0)

        self.input_projection = nn.LazyLinear(self.output_dim)

    def forward(self, batch: dict) -> torch.Tensor:
        pieces = []
        for group_name in self.numeric_feature_groups:
            x = batch["features"].get(group_name)
            if x is not None:
                pieces.append(x.float())

        if not pieces:
            raise ValueError("Model received no feature groups.")

        return self.input_projection(torch.cat(pieces, dim=-1))

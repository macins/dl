from __future__ import annotations

from typing import Mapping, Sequence

from .base import BaseModel
from .registry import register_model
from ..utils.externals import ensure_torch

torch = ensure_torch()
nn = torch.nn


@register_model("gru_sequence_regressor")
class GRUSequenceRegressor(BaseModel):
    def __init__(
        self,
        *,
        numeric_feature_groups: Sequence[str] = ("continuous",),
        categorical_group_name: str = "category",
        categorical_cols: Sequence[str] = (),
        vocab_sizes: Mapping[str, int] | None = None,
        categorical_embedding_dim: int = 8,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        target_key: str = "ret_30min",
    ) -> None:
        super().__init__()
        self.numeric_feature_groups = [str(v) for v in numeric_feature_groups]
        self.categorical_group_name = str(categorical_group_name)
        self.categorical_cols = [str(v) for v in categorical_cols]
        self.target_key = str(target_key)

        self.category_embeddings = nn.ModuleDict()
        for col in self.categorical_cols:
            vocab_size = int((vocab_sizes or {}).get(col, 2))
            self.category_embeddings[col] = nn.Embedding(vocab_size, categorical_embedding_dim, padding_idx=0)

        self.input_projection = nn.LazyLinear(hidden_size)
        self.grus = nn.ModuleList([
            nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=1,
                dropout=0.0,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout)
            for _ in range(num_layers)
        ])
        self.lns = nn.ModuleList([
            nn.LayerNorm(hidden_size)
            for _ in range(num_layers)
        ])
        self.out_ln = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 1),
        )

    def forward(self, batch: dict) -> dict:
        pieces = []
        for group_name in self.numeric_feature_groups:
            x = batch["features"].get(group_name)
            if x is not None:
                pieces.append(x.float())

        if self.categorical_cols:
            x_cat = batch["features"].get(self.categorical_group_name)
            if x_cat is not None:
                cat_embeds = []
                for idx, col in enumerate(self.categorical_cols):
                    cat_embeds.append(self.category_embeddings[col](x_cat[..., idx].long()))
                pieces.append(torch.cat(cat_embeds, dim=-1))

        if not pieces:
            raise ValueError("Model received no feature groups.")

        x = torch.cat(pieces, dim=-1)
        x = self.input_projection(x)
        for i in range(len(self.grus)):
            h, _ = self.grus[i](self.lns[i](x))
            x = self.dropouts[i](h) + x
        pred = self.head(self.out_ln(x))
        return {"preds": {self.target_key: pred}}


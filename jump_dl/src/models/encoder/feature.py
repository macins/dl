from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..base import BaseEncoder
from .registry import register_encoder
from ...utils.externals import ensure_torch

torch = ensure_torch()
nn = torch.nn
Fnn = torch.nn.functional


class DepthwiseCausalEMABank(nn.Module):
    """
    Fully learnable depthwise causal Conv1d EMA-like bank.

    Supports:
        x: (B, T, F)
        x: (B, N, T, F)
        x: (..., T, F)

    Output:
        y: (..., T, F * len(spans))

    Important:
        groups=num_features, so each feature has its own temporal filter.
        No cross-feature interaction happens inside this module.
    """

    def __init__(
        self,
        *,
        num_features: int,
        spans: Sequence[int] = (4, 8, 16, 32, 64),
        trainable: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.num_features = int(num_features)
        self.spans = [int(s) for s in spans]

        if self.num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}.")

        if any(s <= 0 for s in self.spans):
            raise ValueError(f"All spans must be positive, got {self.spans}.")

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=self.num_features,
                    out_channels=self.num_features,
                    kernel_size=span,
                    groups=self.num_features,
                    bias=bias,
                    padding=0,
                )
                for span in self.spans
            ]
        )

        self.reset_parameters_as_ema()

        for p in self.parameters():
            p.requires_grad_(trainable)

    @torch.no_grad()
    def reset_parameters_as_ema(self) -> None:
        for conv, span in zip(self.convs, self.spans):
            alpha = 2.0 / (span + 1.0)

            # Conv1d weight convention after left padding:
            # weight[..., 0]  sees oldest x_{t-span+1}
            # weight[..., -1] sees current x_t
            lags = torch.arange(
                span - 1,
                -1,
                -1,
                dtype=conv.weight.dtype,
                device=conv.weight.device,
            )
            kernel = alpha * (1.0 - alpha) ** lags
            kernel = kernel / kernel.sum()

            conv.weight.copy_(kernel.view(1, 1, span).repeat(self.num_features, 1, 1))

            if conv.bias is not None:
                conv.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 3:
            raise ValueError(
                f"DepthwiseCausalEMABank expects x with shape (..., T, F), "
                f"got {tuple(x.shape)}."
            )

        if x.shape[-1] != self.num_features:
            raise ValueError(
                f"Expected num_features={self.num_features}, "
                f"got input feature dim={x.shape[-1]}."
            )

        leading_shape = x.shape[:-2]
        T = x.shape[-2]
        F = x.shape[-1]

        # (..., T, F) -> (M, T, F)
        x_flat = x.reshape(-1, T, F)

        # (M, T, F) -> (M, F, T)
        x_ch = x_flat.transpose(1, 2)

        outputs = []
        for conv, span in zip(self.convs, self.spans):
            # left padding only; no future leakage
            y = conv(Fnn.pad(x_ch, (span - 1, 0)))

            # (M, F, T) -> (M, T, F)
            outputs.append(y.transpose(1, 2))

        y_flat = torch.cat(outputs, dim=-1)

        # (M, T, F * K) -> (..., T, F * K)
        return y_flat.reshape(*leading_shape, T, self.num_features * len(self.spans))


@register_encoder("tabular_sequence")
class TabularSequenceEncoder(BaseEncoder):
    def __init__(
        self,
        *,
        # own symbol features: these go through raw + EMA
        numeric_feature_groups: Sequence[str] = ("continuous",),

        # context features: these do NOT go through EMA bank
        context_feature_groups: Sequence[str] = ("market_state",),
        use_context_film: bool = False,
        context_hidden_dim: int = 128,
        context_film_scale: float = 0.1,

        categorical_group_name: str = "categorical",
        categorical_cols: Sequence[str] = (),
        vocab_sizes: Mapping[str, int] | None = None,
        categorical_embedding_dim: int = 8,

        model_dim: int = 128,

        # own numeric shape
        num_features: int = 65,

        # EMA bank
        causal_cnn_spans: Sequence[int] = (4, 8, 16, 32, 64),
        causal_cnn_trainable: bool = True,
        include_raw_numeric: bool = True,
        include_categorical_in_own: bool = True,
    ) -> None:
        super().__init__()

        self.numeric_feature_groups = [str(v) for v in numeric_feature_groups]
        self.context_feature_groups = [str(v) for v in context_feature_groups]

        self.categorical_group_name = str(categorical_group_name)
        self.categorical_cols = [str(v) for v in categorical_cols]

        self.num_features = int(num_features)
        self.output_dim = int(model_dim)
        self.include_raw_numeric = bool(include_raw_numeric)
        self.include_categorical_in_own = bool(include_categorical_in_own)

        self.use_context_film = bool(use_context_film)
        self.context_hidden_dim = int(context_hidden_dim)
        self.context_film_scale = float(context_film_scale)

        self.causal_cnn_spans = [int(v) for v in causal_cnn_spans]
        self.categorical_embedding_dim = int(categorical_embedding_dim)

        self.category_embeddings = nn.ModuleDict()

        default_vocab_sizes = {
            "Symbol": 52,
            "DATA_sector": 8,
            "DATA_month": 14,
        }
        merged_vocab_sizes = dict(default_vocab_sizes)
        if vocab_sizes is not None:
            merged_vocab_sizes.update({str(k): int(v) for k, v in vocab_sizes.items()})

        for col in self.categorical_cols:
            vocab_size = merged_vocab_sizes.get(col)
            if vocab_size is None:
                raise ValueError(
                    f"Missing vocab size for categorical column {col!r}. "
                    f"Pass vocab_sizes={{'{col}': ...}}."
                )

            self.category_embeddings[col] = nn.Embedding(
                num_embeddings=int(vocab_size),
                embedding_dim=self.categorical_embedding_dim,
                padding_idx=0,
            )

        self.causal_ema_bank = DepthwiseCausalEMABank(
            num_features=self.num_features,
            spans=self.causal_cnn_spans,
            trainable=causal_cnn_trainable,
            bias=False,
        )

        own_input_dim = self.num_features * len(self.causal_cnn_spans)

        if self.include_raw_numeric:
            own_input_dim += self.num_features

        self.own_projection = nn.Linear(own_input_dim, self.output_dim)

        # context dim may depend on generated market_state / peer_state columns,
        # so keep LazyLinear here. Your trainer EMA has already been changed to
        # state_dict-based EMA, so LazyLinear is now safe.

    @staticmethod
    def _check_same_layout(reference: torch.Tensor, x: torch.Tensor, *, name: str) -> None:
        if x.ndim != reference.ndim:
            raise ValueError(
                f"{name} ndim mismatch: expected ndim={reference.ndim}, got {x.ndim}. "
                f"reference shape={tuple(reference.shape)}, {name} shape={tuple(x.shape)}."
            )

        if x.shape[:-1] != reference.shape[:-1]:
            raise ValueError(
                f"{name} leading/time shape mismatch: expected {tuple(reference.shape[:-1])}, "
                f"got {tuple(x.shape[:-1])}."
            )

    def _collect_numeric(self, batch: dict) -> torch.Tensor:
        pieces = []

        features = batch.get("features", {})
        for group_name in self.numeric_feature_groups:
            x = features.get(group_name)
            if x is not None:
                pieces.append(x.float())

        if not pieces:
            raise ValueError(
                f"Model received no own numeric feature groups. "
                f"Expected one of {self.numeric_feature_groups}."
            )

        x_numeric = torch.cat(pieces, dim=-1)

        if x_numeric.ndim not in (3, 4):
            raise ValueError(
                f"Numeric features should have shape (B, T, F) or (B, N, T, F), "
                f"got {tuple(x_numeric.shape)}."
            )

        if x_numeric.shape[-1] != self.num_features:
            raise ValueError(
                f"Expected total own numeric features = {self.num_features}, "
                f"but got {x_numeric.shape[-1]}. "
                f"Check numeric_feature_groups or num_features."
            )

        return x_numeric

    def _collect_context(self, batch: dict, reference: torch.Tensor) -> torch.Tensor | None:
        if not self.context_feature_groups:
            return None

        features = batch.get("features", {})
        pieces = []

        for group_name in self.context_feature_groups:
            x = features.get(group_name)
            if x is None:
                continue

            x = x.float()

            if x.ndim not in (3, 4):
                raise ValueError(
                    f"Context group {group_name!r} should have shape "
                    f"(B, T, C) or (B, N, T, C), got {tuple(x.shape)}."
                )

            self._check_same_layout(reference, x, name=f"context group {group_name!r}")
            pieces.append(x)

        if not pieces:
            return None

        return torch.cat(pieces, dim=-1)

    def _collect_categorical_embeddings(
        self,
        batch: dict,
        reference: torch.Tensor,
    ) -> list[torch.Tensor]:
        if not self.categorical_cols:
            return []

        features = batch.get("features", {})
        cat_group = features.get(self.categorical_group_name)
        if cat_group is None:
            return []

        outputs = []

        for i, col in enumerate(self.categorical_cols):
            emb = self.category_embeddings[col]

            if isinstance(cat_group, Mapping):
                idx = cat_group.get(col)
                if idx is None:
                    continue
            else:
                # cat_group:
                #   (B, T, C) or (B, N, T, C)
                idx = cat_group[..., i]

            if idx.ndim >= 1 and idx.shape[-1:] == (1,):
                idx = idx.squeeze(-1)

            # idx should be:
            #   (B, T) or (B, N, T)
            expected_shape = reference.shape[:-1]
            if idx.shape != expected_shape:
                raise ValueError(
                    f"Categorical column {col!r} shape mismatch: "
                    f"expected {tuple(expected_shape)}, got {tuple(idx.shape)}."
                )

            idx = idx.long()
            outputs.append(emb(idx))

        return outputs

    def forward(self, batch: dict) -> torch.Tensor:
        x_numeric = self._collect_numeric(batch)

        own_pieces = []

        if self.include_raw_numeric:
            own_pieces.append(x_numeric)

        x_ema = self.causal_ema_bank(x_numeric)
        own_pieces.append(x_ema)

        if self.include_categorical_in_own:
            own_pieces.extend(
                self._collect_categorical_embeddings(batch, reference=x_numeric)
            )

        x_own = torch.cat(own_pieces, dim=-1)
        h = self.own_projection(x_own)
        return h

        x_ctx = self._collect_context(batch, reference=x_numeric)

        if x_ctx is None:
            return h

        if self.use_context_film:
            gamma_beta = self.context_mlp(x_ctx)
            gamma, beta = gamma_beta.chunk(2, dim=-1)

            scale = self.context_film_scale
            h = h * (1.0 + scale * torch.tanh(gamma)) + scale * beta
            return h

        h_ctx = self.context_concat_projection(x_ctx)
        return h + h_ctx
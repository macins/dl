from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .norms import build_norm
from .registry import register_block
from ...utils.externals import ensure_torch

torch = ensure_torch()
nn = torch.nn
F = nn.functional


def _build_position_ids(
    *,
    leading_shape: tuple[int, ...],
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Return position ids with shape:

        (*leading_shape, seq_len)

    Examples:
        leading_shape=(B,)    -> (B, T)
        leading_shape=(B, N)  -> (B, N, T)
    """
    pos = torch.arange(seq_len, device=device)
    view_shape = (1,) * len(leading_shape) + (seq_len,)
    return pos.view(*view_shape).expand(*leading_shape, seq_len)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def _apply_rope(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    base: float = 10000.0,
) -> torch.Tensor:
    """
    x:
        (..., num_heads, seq_len, head_dim)

    position_ids:
        (..., seq_len)

    Output:
        same shape as x
    """
    head_dim = x.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires even head_dim, got {head_dim}")

    freq_seq = torch.arange(
        0,
        head_dim,
        2,
        device=x.device,
        dtype=torch.float32,
    )
    inv_freq = 1.0 / (float(base) ** (freq_seq / head_dim))

    angles = position_ids.to(dtype=torch.float32).unsqueeze(-1) * inv_freq.view(
        *([1] * position_ids.ndim),
        -1,
    )

    cos = torch.repeat_interleave(torch.cos(angles), repeats=2, dim=-1).unsqueeze(-3)
    sin = torch.repeat_interleave(torch.sin(angles), repeats=2, dim=-1).unsqueeze(-3)

    x_float = x.float()
    return (x_float * cos + _rotate_half(x_float) * sin).to(dtype=x.dtype)


def _build_alibi_bias(
    num_heads: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Return:
        (1, num_heads, seq_len, seq_len)

    Broadcasts to:
        (B, num_heads, T, T)
        (B, N, num_heads, T, T)
    """
    slopes = torch.tensor(
        [2.0 ** (-(8.0 * (i + 1) / num_heads)) for i in range(num_heads)],
        device=device,
        dtype=torch.float32,
    ).view(1, num_heads, 1, 1)

    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    distance = (
        positions.view(1, 1, seq_len, 1)
        - positions.view(1, 1, 1, seq_len)
    ).abs()

    bias = -slopes * distance
    return bias.to(dtype=dtype)


def _build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Return bool mask where True means masked / forbidden.
    """
    return torch.triu(
        torch.ones((seq_len, seq_len), device=device, dtype=torch.bool),
        diagonal=1,
    )


class PositionEmbedding(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        max_seq_len: int = 4096,
        mode: str = "rope",
    ) -> None:
        super().__init__()

        self.model_dim = int(model_dim)
        self.max_seq_len = int(max_seq_len)
        self.mode = str(mode).strip().lower()

        if self.mode == "absolute":
            self.embedding = nn.Embedding(self.max_seq_len, self.model_dim)
        else:
            self.embedding = None

    def add_to_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:
            (..., T, D)
        """
        if self.mode != "absolute":
            return x

        if x.ndim < 3:
            raise ValueError(
                f"PositionEmbedding expects x with shape (..., T, D), got {tuple(x.shape)}"
            )

        seq_len = x.shape[-2]
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        pos = torch.arange(seq_len, device=x.device)
        emb = self.embedding(pos)  # (T, D)

        view_shape = (1,) * (x.ndim - 2) + emb.shape
        return x + emb.view(*view_shape)

    def apply_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        q/k:
            (..., H, T, Hd)
        """
        if self.mode != "rope":
            return q, k

        leading_shape = tuple(q.shape[:-3])
        seq_len = q.shape[-2]
        pos = _build_position_ids(
            leading_shape=leading_shape,
            seq_len=seq_len,
            device=q.device,
        )

        return _apply_rope(q, pos), _apply_rope(k, pos)


class MultiHeadSelfAttention(nn.Module):
    """
    Temporal self-attention over the last sequence dimension T.

    Supports:
        x: (B, T, D)
        x: (B, N, T, D)

    Uses PyTorch SDPA:
        F.scaled_dot_product_attention(...)
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        qk_norm: bool = True,
        attention_gate: bool = True,
        norm_type: str = "rmsnorm",
        max_seq_len: int = 4096,
        position_encoding: str = "rope",
        causal: bool = False,
    ) -> None:
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}"
            )

        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.qk_norm = bool(qk_norm)
        self.use_attention_gate = bool(attention_gate)
        self.causal = bool(causal)

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.q_norm = build_norm(norm_type, self.head_dim) if self.qk_norm else nn.Identity()
        self.k_norm = build_norm(norm_type, self.head_dim) if self.qk_norm else nn.Identity()

        self.gate_proj = (
            nn.Linear(self.hidden_size, self.hidden_size)
            if self.use_attention_gate
            else None
        )

        self.position = PositionEmbedding(
            model_dim=self.hidden_size,
            max_seq_len=max_seq_len,
            mode=position_encoding,
        )

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:
            (..., T, D)

        return:
            (..., H, T, Hd)
        """
        if x.ndim < 3:
            raise ValueError(
                f"MultiHeadSelfAttention expects x with shape (..., T, D), "
                f"got {tuple(x.shape)}."
            )

        leading_shape = x.shape[:-2]
        seq_len = x.shape[-2]
        hidden = x.shape[-1]

        if hidden != self.hidden_size:
            raise ValueError(
                f"Expected hidden_size={self.hidden_size}, got {hidden}."
            )

        x = x.view(*leading_shape, seq_len, self.num_heads, self.head_dim)
        return x.transpose(-3, -2).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:
            (..., H, T, Hd)

        return:
            (..., T, D)
        """
        leading_shape = x.shape[:-3]
        seq_len = x.shape[-2]

        x = x.transpose(-3, -2).contiguous()
        return x.view(*leading_shape, seq_len, self.hidden_size)

    def _build_sdpa_mask(
        self,
        *,
        q: torch.Tensor,
        padding_mask: torch.Tensor | None,
        seq_len: int,
    ) -> tuple[torch.Tensor | None, bool]:
        """
        Return:
            attn_mask:
                bool mask where True means allowed,
                or float additive bias mask,
                or None.

            is_causal:
                bool flag for SDPA.

        q:
            (..., H, T, Hd)

        SDPA expects mask broadcastable to:
            (..., H, T_query, T_key)
        """
        allowed_mask: torch.Tensor | None = None

        if padding_mask is not None:
            # padding_mask: (..., T), True means valid.
            #
            # key_valid_safe avoids all-False key rows, which can produce NaNs.
            # Invalid query positions are zeroed after attention.
            key_valid = padding_mask.bool()
            has_any_key = key_valid.any(dim=-1, keepdim=True)
            key_valid_safe = torch.where(has_any_key, key_valid, torch.ones_like(key_valid))

            # (..., 1, 1, T_key)
            allowed_mask = key_valid_safe.unsqueeze(-2).unsqueeze(-3)

        use_sdpa_is_causal = False

        if self.causal:
            causal_allowed = ~_build_causal_mask(seq_len, q.device)
            causal_allowed = causal_allowed.view(
                *([1] * (q.ndim - 2)),
                seq_len,
                seq_len,
            )

            if allowed_mask is None:
                allowed_mask = causal_allowed
            else:
                allowed_mask = allowed_mask & causal_allowed

        if self.position.mode == "alibi":
            alibi = _build_alibi_bias(
                num_heads=self.num_heads,
                seq_len=seq_len,
                device=q.device,
                dtype=q.dtype,
            )

            if allowed_mask is None:
                return alibi, False

            # Convert bool allowed mask into additive bias with ALiBi.
            neg_inf = torch.finfo(q.dtype).min
            attn_mask = alibi.masked_fill(~allowed_mask, neg_inf)
            return attn_mask, False

        if allowed_mask is None:
            use_sdpa_is_causal = self.causal
            return None, use_sdpa_is_causal

        return allowed_mask, False

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x:
            (B, T, D)
            or (B, N, T, D)

        padding_mask:
            (B, T)
            or (B, N, T)

        Attention is always over the last sequence dimension T.
        """
        if x.ndim not in (3, 4):
            raise ValueError(
                f"MultiHeadSelfAttention supports (B,T,D) or (B,N,T,D), "
                f"got {tuple(x.shape)}."
            )

        if padding_mask is not None:
            expected_mask_shape = x.shape[:-1]
            if padding_mask.shape != expected_mask_shape:
                raise ValueError(
                    f"padding_mask shape mismatch: expected {tuple(expected_mask_shape)}, "
                    f"got {tuple(padding_mask.shape)}."
                )

        x = self.position.add_to_input(x)

        q = self._reshape_heads(self.q_proj(x))
        k = self._reshape_heads(self.k_proj(x))
        v = self._reshape_heads(self.v_proj(x))

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k = self.position.apply_qk(q, k)

        seq_len = x.shape[-2]
        attn_mask, is_causal = self._build_sdpa_mask(
            q=q,
            padding_mask=padding_mask,
            seq_len=seq_len,
        )

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal,
            scale=self.scale,
        )

        out = self._merge_heads(out)

        if self.gate_proj is not None:
            out = out * torch.sigmoid(self.gate_proj(x))

        out = self.out_proj(out)

        if padding_mask is not None:
            out = out * padding_mask.unsqueeze(-1).to(dtype=out.dtype)

        return out


class CrossSymbolAttention(nn.Module):
    """
    Cross-symbol self-attention at each timestamp.

    Input:
        x: (B, N, T, D)

    Output:
        y: (B, N, T, D)

    Semantics:
        For each (B, T), attend over N symbols.

    padding_mask:
        (B, N, T), True means valid.

    Symbol embedding:
        Adds learned slot embedding e_n to x[:, n, :, :].

    Uses PyTorch SDPA:
        F.scaled_dot_product_attention(...)
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        qk_norm: bool = True,
        attention_gate: bool = True,
        norm_type: str = "rmsnorm",
        max_symbols: int = 512,
        use_symbol_embedding: bool = True,
        symbol_embedding_scale: float = 1.0,
        mask_self: bool = False,
        num_register_tokens: int = 0,
    ) -> None:
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}"
            )

        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.qk_norm = bool(qk_norm)
        self.use_attention_gate = bool(attention_gate)

        self.max_symbols = int(max_symbols)
        self.use_symbol_embedding = bool(use_symbol_embedding)
        self.symbol_embedding_scale = float(symbol_embedding_scale)
        self.mask_self = bool(mask_self)
        self.num_register_tokens = int(num_register_tokens)
        if self.num_register_tokens < 0:
            raise ValueError(
                f"num_register_tokens must be non-negative, got {self.num_register_tokens}"
            )

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.q_norm = build_norm(norm_type, self.head_dim) if self.qk_norm else nn.Identity()
        self.k_norm = build_norm(norm_type, self.head_dim) if self.qk_norm else nn.Identity()

        self.gate_proj = (
            nn.Linear(self.hidden_size, self.hidden_size)
            if self.use_attention_gate
            else None
        )

        self.symbol_embedding = (
            nn.Embedding(self.max_symbols, self.hidden_size)
            if self.use_symbol_embedding
            else None
        )

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:
            (B, T, N, D)

        return:
            (B, T, H, N, Hd)
        """
        B, T, N, D = x.shape
        if D != self.hidden_size:
            raise ValueError(f"Expected hidden_size={self.hidden_size}, got {D}.")

        x = x.view(B, T, N, self.num_heads, self.head_dim)
        return x.permute(0, 1, 3, 2, 4).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:
            (B, T, H, N, Hd)

        return:
            (B, T, N, D)
        """
        B, T, H, N, Hd = x.shape
        return x.permute(0, 1, 3, 2, 4).contiguous().view(B, T, N, self.hidden_size)

    def _add_symbol_embedding(self, x_btnd: torch.Tensor) -> torch.Tensor:
        """
        x_btnd:
            (B, T, N_total, D)

        If num_register_tokens > 0, the first R slots are treated as register
        tokens and do not receive ordinary symbol embeddings. This avoids
        shifting the learned symbol ids of the real symbols when register tokens
        are prepended along the symbol axis.
        """
        if self.symbol_embedding is None:
            return x_btnd

        N_total = x_btnd.shape[2]
        R = min(self.num_register_tokens, N_total)
        N_symbols = N_total - R

        if N_symbols <= 0:
            return x_btnd

        if N_symbols > self.max_symbols:
            raise ValueError(
                f"N_symbols={N_symbols} exceeds max_symbols={self.max_symbols}. "
                f"Increase max_symbols in cross_symbol_attention config."
            )

        symbol_ids = torch.arange(N_symbols, device=x_btnd.device)
        emb = self.symbol_embedding(symbol_ids)  # (N_symbols, D)

        out = x_btnd.clone()
        out[:, :, R:, :] = out[:, :, R:, :] + self.symbol_embedding_scale * emb.view(
            1, 1, N_symbols, self.hidden_size
        )
        return out

    def _build_symbol_allowed_mask(
        self,
        *,
        padding_mask: torch.Tensor | None,
        B: int,
        T: int,
        N: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """
        Return bool mask broadcastable to:
            (B, T, H, N_query, N_key)

        True means allowed.
        """
        allowed: torch.Tensor | None = None

        if padding_mask is not None:
            # padding_mask: (B, N, T) -> (B, T, N)
            mask_btn = padding_mask.bool().transpose(1, 2).contiguous()

            # key_allowed: (B, T, 1, 1, N_key)
            key_allowed = mask_btn.unsqueeze(2).unsqueeze(3)

            # Expand to query dimension:
            # (B, T, 1, N_query, N_key)
            allowed = key_allowed.expand(B, T, 1, N, N)

        if self.mask_self and N > 1:
            not_self = ~torch.eye(N, device=device, dtype=torch.bool).view(1, 1, 1, N, N)
            allowed = not_self if allowed is None else (allowed & not_self)

        if allowed is not None:
            # Avoid all-False rows, which can produce NaNs in SDPA.
            # If a query has no allowed key, fall back to allowing all keys;
            # output is still zeroed afterwards for invalid queries.
            row_has_key = allowed.any(dim=-1, keepdim=True)
            allowed = torch.where(row_has_key, allowed, torch.ones_like(allowed))

        return allowed

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"CrossSymbolAttention expects x with shape (B,N,T,D), got {tuple(x.shape)}."
            )

        B, N, T, D = x.shape
        if D != self.hidden_size:
            raise ValueError(f"Expected hidden_size={self.hidden_size}, got {D}.")

        if padding_mask is not None and padding_mask.shape != (B, N, T):
            raise ValueError(
                f"padding_mask shape mismatch: expected {(B, N, T)}, "
                f"got {tuple(padding_mask.shape)}."
            )

        # (B, N, T, D) -> (B, T, N, D)
        x_btnd = x.transpose(1, 2).contiguous()
        x_btnd = self._add_symbol_embedding(x_btnd)

        q = self._reshape_heads(self.q_proj(x_btnd))
        k = self._reshape_heads(self.k_proj(x_btnd))
        v = self._reshape_heads(self.v_proj(x_btnd))

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        attn_mask = self._build_symbol_allowed_mask(
            padding_mask=padding_mask,
            B=B,
            T=T,
            N=N,
            device=x.device,
        )

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
            scale=self.scale,
        )

        out = self._merge_heads(out)

        if self.gate_proj is not None:
            out = out * torch.sigmoid(self.gate_proj(x_btnd))

        out = self.out_proj(out)

        # (B, T, N, D) -> (B, N, T, D)
        out = out.transpose(1, 2).contiguous()

        if padding_mask is not None:
            out = out * padding_mask.unsqueeze(-1).to(dtype=out.dtype)

        return out


class FeedForward(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        ffn_hidden_size: int | None = None,
        activation: str = "swiglu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.hidden_size = int(hidden_size)
        self.ffn_hidden_size = int(ffn_hidden_size or 2 * hidden_size)

        self.activation = str(activation).strip().lower()
        self.is_gated = self.activation in {"glu", "geglu", "swiglu"}

        inner_dim = self.ffn_hidden_size * (2 if self.is_gated else 1)

        self.in_proj = nn.Linear(self.hidden_size, inner_dim)
        self.out_proj = nn.Linear(self.ffn_hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(dropout)

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "gelu":
            return F.gelu(x)
        if self.activation == "silu":
            return F.silu(x)
        if self.activation == "glu":
            a, gate = x.chunk(2, dim=-1)
            return a * torch.sigmoid(gate)
        if self.activation == "geglu":
            a, gate = x.chunk(2, dim=-1)
            return a * F.gelu(gate)
        if self.activation == "swiglu":
            a, gate = x.chunk(2, dim=-1)
            return a * F.silu(gate)

        raise KeyError(f"Unknown FFN activation: {self.activation}")

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self._activate(self.in_proj(x))
        out = self.out_proj(self.dropout(out))

        if padding_mask is not None:
            out = out * padding_mask.unsqueeze(-1).to(dtype=out.dtype)

        return out


class MoEFeedForward(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        dense_ffn_hidden_size: int,
        expert_hidden_size: int | None = None,
        num_experts: int = 4,
        top_k: int = 2,
        activation: str = "swiglu",
        dropout: float = 0.1,
        aux_loss_weight: float = 1e-2,
        router_z_loss_weight: float = 0,
        shared_experts: int = 0,
    ) -> None:
        super().__init__()

        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")

        self.hidden_size = int(hidden_size)
        self.dense_ffn_hidden_size = int(dense_ffn_hidden_size)

        default_expert_hidden = max(1, self.dense_ffn_hidden_size // top_k)
        self.expert_hidden_size = int(expert_hidden_size or default_expert_hidden)

        if self.expert_hidden_size >= self.dense_ffn_hidden_size:
            raise ValueError(
                "expert_hidden_size must be strictly smaller than dense_ffn_hidden_size "
                f"(got expert_hidden_size={self.expert_hidden_size}, "
                f"dense_ffn_hidden_size={self.dense_ffn_hidden_size})"
            )

        self.num_experts = int(num_experts)
        self.top_k = min(int(top_k), self.num_experts)

        self.aux_loss_weight = float(aux_loss_weight)
        self.router_z_loss_weight = float(router_z_loss_weight)
        self.shared_experts = int(shared_experts)

        self.router = nn.Linear(self.hidden_size, self.num_experts)

        self.experts = nn.ModuleList(
            [
                FeedForward(
                    hidden_size=self.hidden_size,
                    ffn_hidden_size=self.expert_hidden_size,
                    activation=activation,
                    dropout=dropout,
                )
                for _ in range(self.num_experts)
            ]
        )

        self.shared_expert_layers = nn.ModuleList(
            [
                FeedForward(
                    hidden_size=self.hidden_size,
                    ffn_hidden_size=self.expert_hidden_size,
                    activation=activation,
                    dropout=dropout,
                )
                for _ in range(self.shared_experts)
            ]
        )

    def _router_statistics(
        self,
        *,
        router_probs: torch.Tensor,
        expert_selection: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        router_probs:
            (..., T, E)

        expert_selection:
            (..., T, E)

        padding_mask:
            (..., T)

        Return:
            mean_probs:     (E,)
            mean_selection: (E,)
        """
        reduce_dims = tuple(range(router_probs.ndim - 1))

        if padding_mask is not None:
            valid = padding_mask.to(dtype=router_probs.dtype)
            denom = valid.sum().clamp_min(1.0)

            mean_probs = (
                router_probs * valid.unsqueeze(-1)
            ).sum(dim=reduce_dims) / denom

            mean_selection = (
                expert_selection * valid.unsqueeze(-1)
            ).sum(dim=reduce_dims) / (denom * self.top_k)
        else:
            mean_probs = router_probs.mean(dim=reduce_dims)
            mean_selection = expert_selection.mean(dim=reduce_dims) / self.top_k

        return mean_probs, mean_selection

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
        """
        x:
            (B, T, D)
            or (B, N, T, D)
        """
        router_logits = self.router(x)
        router_probs = torch.softmax(router_logits, dim=-1)

        topk_values, topk_indices = torch.topk(
            router_probs,
            k=self.top_k,
            dim=-1,
        )

        topk_values = topk_values / topk_values.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-9)

        expert_outputs = torch.stack(
            [expert(x, padding_mask=padding_mask) for expert in self.experts],
            dim=-2,
        )

        gather_index = topk_indices.unsqueeze(-1).expand(
            *topk_indices.shape,
            self.hidden_size,
        )

        selected_outputs = expert_outputs.gather(dim=-2, index=gather_index)
        out = (selected_outputs * topk_values.unsqueeze(-1)).sum(dim=-2)

        if self.shared_expert_layers:
            shared_out = torch.stack(
                [
                    expert(x, padding_mask=padding_mask)
                    for expert in self.shared_expert_layers
                ],
                dim=0,
            ).mean(dim=0)

            out = out + shared_out

        expert_selection = torch.zeros_like(router_probs).scatter(
            -1,
            topk_indices,
            1.0,
        )

        mean_probs, mean_selection = self._router_statistics(
            router_probs=router_probs,
            expert_selection=expert_selection,
            padding_mask=padding_mask,
        )

        aux_loss = (
            self.aux_loss_weight
            * self.num_experts
            * torch.sum(mean_probs * mean_selection)
        )

        router_z_loss = self.router_z_loss_weight * torch.mean(
            torch.logsumexp(router_logits, dim=-1).pow(2)
        )

        metrics = {
            "moe_aux_loss": float(aux_loss.detach().item()),
            "moe_router_z_loss": float(router_z_loss.detach().item()),
            "moe_expert_usage_max": float(mean_selection.max().detach().item()),
            "moe_expert_usage_min": float(mean_selection.min().detach().item()),
        }

        losses = {
            "aux_loss": aux_loss,
            "router_z_loss": router_z_loss,
        }

        return out, losses, metrics


def _normalize_sublayer_specs(
    sublayers: Sequence[str | Mapping[str, Any]] | None,
    *,
    use_moe: bool,
) -> list[dict[str, Any]]:
    """
    Convert YAML-friendly sublayer specs into normalized dicts.

    Supported forms:

        sublayers:
          - attention
          - ffn

    or:

        sublayers:
          - type: attention
            causal: true
          - type: ffn
            ffn_hidden_size: 512
          - type: moe_ffn
            num_experts: 4
            top_k: 2
          - type: cross_symbol_attention
            max_symbols: 128
    """
    if sublayers is None:
        return [
            {"type": "attention"},
            {"type": "moe_ffn" if use_moe else "ffn"},
        ]

    out: list[dict[str, Any]] = []

    for i, item in enumerate(sublayers):
        if isinstance(item, str):
            spec = {"type": item}
        elif isinstance(item, Mapping):
            spec = dict(item)
        else:
            raise TypeError(
                f"sublayers[{i}] must be a string or mapping, got {type(item)!r}"
            )

        layer_type = str(
            spec.pop("type", spec.pop("kind", spec.pop("name", "")))
        ).strip().lower()

        aliases = {
            "attn": "attention",
            "self_attention": "attention",
            "self-attention": "attention",
            "mha": "attention",

            "cross_attention": "cross_symbol_attention",
            "cross-attention": "cross_symbol_attention",
            "cross_symbol_attn": "cross_symbol_attention",
            "cross-symbol-attention": "cross_symbol_attention",
            "symbol_attention": "cross_symbol_attention",
            "symbol_attn": "cross_symbol_attention",

            "mlp": "ffn",
            "feedforward": "ffn",
            "feed_forward": "ffn",
            "dense_ffn": "ffn",

            "moe": "moe_ffn",
            "moeffn": "moe_ffn",
            "moe-feedforward": "moe_ffn",
            "moe_feedforward": "moe_ffn",
        }

        layer_type = aliases.get(layer_type, layer_type)

        if layer_type not in {"attention", "cross_symbol_attention", "ffn", "moe_ffn"}:
            raise ValueError(
                f"Unknown sublayer type {layer_type!r}. "
                "Supported: attention, cross_symbol_attention, ffn, moe_ffn."
            )

        spec["type"] = layer_type
        out.append(spec)

    if not out:
        raise ValueError("sublayers cannot be empty.")

    return out


class _ResidualSubLayer(nn.Module):
    """
    One PreNorm residual sublayer.

    Supports:
        - attention
        - cross_symbol_attention
        - ffn
        - moe_ffn

    Input:
        x: (B, T, D) or (B, N, T, D)

    padding_mask:
        (B, T) or (B, N, T)
    """

    def __init__(
        self,
        *,
        layer_type: str,
        module: nn.Module,
        hidden_size: int,
        norm_type: str,
        norm_eps: float,
        dropout: float,
        residual_scale: float,
    ) -> None:
        super().__init__()

        self.layer_type = str(layer_type)
        self.module = module
        self.norm = build_norm(norm_type, hidden_size, eps=norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = float(residual_scale)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
        losses: dict[str, torch.Tensor] = {}
        metrics: dict[str, float] = {}

        y = self.norm(x)

        if self.layer_type == "attention":
            out = self.module(y, padding_mask=padding_mask)

        elif self.layer_type == "cross_symbol_attention":
            out = self.module(y, padding_mask=padding_mask)

        elif self.layer_type == "ffn":
            out = self.module(y, padding_mask=padding_mask)

        elif self.layer_type == "moe_ffn":
            out, losses, metrics = self.module(y, padding_mask=padding_mask)

        else:
            raise RuntimeError(f"Unexpected layer_type={self.layer_type!r}")

        x = x + self.residual_scale * self.dropout(out)

        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1).to(dtype=x.dtype)

        return x, losses, metrics


@register_block("transformer")
@register_block("transformer_encoder")
class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_hidden_size: int | None = None,
        ffn_activation: str = "swiglu",
        norm_type: str = "rmsnorm",
        norm_eps: float = 1e-6,
        residual_scale: float = 1.0,
        qk_norm: bool = True,
        attention_gate: bool = True,
        max_seq_len: int = 4096,
        position_encoding: str = "rope",
        causal: bool = False,

        use_moe: bool = False,
        num_experts: int = 4,
        expert_hidden_size: int | None = None,
        top_k: int = 2,
        aux_loss_weight: float = 1e-2,
        router_z_loss_weight: float = 1e-3,
        shared_experts: int = 0,

        sublayers: Sequence[str | Mapping[str, Any]] | None = None,

        use_register_tokens: bool = False,
        num_register_tokens: int = 1,
        register_axis: str = "symbol",
        return_register_tokens: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_size = int(hidden_size)
        self.use_moe = bool(use_moe)

        self.last_aux_losses: dict[str, torch.Tensor] = {}
        self.last_aux_metrics: dict[str, float] = {}

        self.use_register_tokens = bool(use_register_tokens)
        self.num_register_tokens = int(num_register_tokens)
        self.register_axis = str(register_axis).strip().lower()
        self.return_register_tokens = bool(return_register_tokens)

        if self.use_register_tokens:
            if self.register_axis != "symbol":
                raise ValueError(
                    "Currently only register_axis='symbol' is supported. "
                    "Expected input shape (B,N,T,D)."
                )
            if self.num_register_tokens <= 0:
                raise ValueError(
                    f"num_register_tokens must be positive, got {self.num_register_tokens}"
                )
            self.register_tokens = nn.Parameter(
                torch.zeros(self.num_register_tokens, self.hidden_size)
            )
            nn.init.normal_(self.register_tokens, mean=0.0, std=0.02)
        else:
            self.num_register_tokens = 0
            self.register_tokens = None

        dense_ffn_hidden_size = int(ffn_hidden_size or 4 * hidden_size)

        specs = _normalize_sublayer_specs(
            sublayers,
            use_moe=self.use_moe,
        )

        layers: list[_ResidualSubLayer] = []

        for i, spec in enumerate(specs):
            spec = dict(spec)
            layer_type = str(spec.pop("type")).strip().lower()

            layer_dropout = float(spec.pop("dropout", dropout))
            layer_residual_scale = float(spec.pop("residual_scale", residual_scale))
            layer_norm_type = str(spec.pop("norm_type", norm_type))
            layer_norm_eps = float(spec.pop("norm_eps", norm_eps))

            if layer_type == "attention":
                module = MultiHeadSelfAttention(
                    hidden_size=self.hidden_size,
                    num_heads=int(spec.pop("num_heads", num_heads)),
                    dropout=layer_dropout,
                    qk_norm=bool(spec.pop("qk_norm", qk_norm)),
                    attention_gate=bool(spec.pop("attention_gate", attention_gate)),
                    norm_type=str(spec.pop("qk_norm_type", norm_type)),
                    max_seq_len=int(spec.pop("max_seq_len", max_seq_len)),
                    position_encoding=str(spec.pop("position_encoding", position_encoding)),
                    causal=bool(spec.pop("causal", causal)),
                )

            elif layer_type == "cross_symbol_attention":
                module = CrossSymbolAttention(
                    hidden_size=self.hidden_size,
                    num_heads=int(spec.pop("num_heads", num_heads)),
                    dropout=layer_dropout,
                    qk_norm=bool(spec.pop("qk_norm", qk_norm)),
                    attention_gate=bool(spec.pop("attention_gate", attention_gate)),
                    norm_type=str(spec.pop("qk_norm_type", norm_type)),
                    max_symbols=int(spec.pop("max_symbols", 512)),
                    use_symbol_embedding=bool(spec.pop("use_symbol_embedding", True)),
                    symbol_embedding_scale=float(spec.pop("symbol_embedding_scale", 1.0)),
                    mask_self=bool(spec.pop("mask_self", False)),
                    num_register_tokens=int(
                        spec.pop("num_register_tokens", self.num_register_tokens)
                    ),
                )

            elif layer_type == "ffn":
                module = FeedForward(
                    hidden_size=self.hidden_size,
                    ffn_hidden_size=int(
                        spec.pop(
                            "ffn_hidden_size",
                            spec.pop("hidden_size_ffn", dense_ffn_hidden_size),
                        )
                    ),
                    activation=str(spec.pop("activation", ffn_activation)),
                    dropout=layer_dropout,
                )

            elif layer_type == "moe_ffn":
                module = MoEFeedForward(
                    hidden_size=self.hidden_size,
                    dense_ffn_hidden_size=int(
                        spec.pop(
                            "dense_ffn_hidden_size",
                            spec.pop("ffn_hidden_size", dense_ffn_hidden_size),
                        )
                    ),
                    expert_hidden_size=spec.pop("expert_hidden_size", expert_hidden_size),
                    num_experts=int(spec.pop("num_experts", num_experts)),
                    top_k=int(spec.pop("top_k", top_k)),
                    activation=str(spec.pop("activation", ffn_activation)),
                    dropout=layer_dropout,
                    aux_loss_weight=float(spec.pop("aux_loss_weight", aux_loss_weight)),
                    router_z_loss_weight=float(
                        spec.pop("router_z_loss_weight", router_z_loss_weight)
                    ),
                    shared_experts=int(spec.pop("shared_experts", shared_experts)),
                )

            else:
                raise RuntimeError(f"Unexpected sublayer type {layer_type!r}")

            if spec:
                raise ValueError(
                    f"Unused keys in sublayers[{i}] ({layer_type}): {sorted(spec.keys())}"
                )

            layers.append(
                _ResidualSubLayer(
                    layer_type=layer_type,
                    module=module,
                    hidden_size=self.hidden_size,
                    norm_type=layer_norm_type,
                    norm_eps=layer_norm_eps,
                    dropout=layer_dropout,
                    residual_scale=layer_residual_scale,
                )
            )

        self.layers = nn.ModuleList(layers)

    def _prepend_register_tokens(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Prepend learnable register tokens along the symbol axis.

        Input:
            x:            (B, N, T, D)
            padding_mask: (B, N, T), True means valid

        Output:
            x:            (B, R + N, T, D)
            padding_mask: (B, R + N, T)

        Because the register tokens are inserted before the sublayer loop, they
        are processed by temporal self-attention, cross-symbol attention, and
        FFN/MoE exactly like ordinary tokens.
        """
        if not self.use_register_tokens:
            return x, padding_mask

        if x.ndim != 4:
            raise ValueError(
                "symbol-axis register tokens require x with shape (B,N,T,D), "
                f"got {tuple(x.shape)}."
            )

        B, _, T, D = x.shape
        R = self.num_register_tokens

        if self.register_tokens is None:
            raise RuntimeError("register_tokens is None while use_register_tokens=True.")

        reg = self.register_tokens.view(1, R, 1, D).expand(B, R, T, D)
        x = torch.cat([reg, x], dim=1)

        if padding_mask is not None:
            reg_mask = torch.ones(
                B,
                R,
                T,
                device=padding_mask.device,
                dtype=torch.bool,
            )
            padding_mask = torch.cat([reg_mask, padding_mask.bool()], dim=1)

        return x, padding_mask

    def _remove_register_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Drop register tokens from the output unless return_register_tokens=True.
        This keeps the default output shape equal to the original input shape.
        """
        if not self.use_register_tokens or self.return_register_tokens:
            return x

        return x[:, self.num_register_tokens :, :, :].contiguous()

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x:
            (B, T, D)
            or (B, N, T, D)

        padding_mask:
            (B, T)
            or (B, N, T)
        """
        if x.ndim not in (3, 4):
            raise ValueError(
                f"TransformerEncoderBlock supports (B,T,D) or (B,N,T,D), "
                f"got {tuple(x.shape)}."
            )

        if padding_mask is not None and padding_mask.shape != x.shape[:-1]:
            raise ValueError(
                f"padding_mask shape mismatch: expected {tuple(x.shape[:-1])}, "
                f"got {tuple(padding_mask.shape)}."
            )

        x, padding_mask = self._prepend_register_tokens(x, padding_mask)

        self.last_aux_losses = {}
        self.last_aux_metrics = {}

        for i, layer in enumerate(self.layers):
            x, losses, metrics = layer(x, padding_mask=padding_mask)

            if losses:
                for name, value in losses.items():
                    self.last_aux_losses[f"layer_{i}_{layer.layer_type}_{name}"] = value

            if metrics:
                for name, value in metrics.items():
                    self.last_aux_metrics[f"layer_{i}_{layer.layer_type}_{name}"] = value

        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1).to(dtype=x.dtype)

        x = self._remove_register_tokens(x)

        return x

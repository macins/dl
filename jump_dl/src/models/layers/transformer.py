from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .norms import build_norm
from .codebook import CodebookAdapter
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
    return torch.triu(
        torch.ones((seq_len, seq_len), device=device, dtype=torch.bool),
        diagonal=1,
    )


def _normalize_attention_projection_sharing(
    mode: str | Sequence[str] | None,
) -> set[str]:
    """
    Supported examples:
        None, "none", "false", ""
        "q", "k", "v"
        "qk", "qv", "kv", "qkv"
        "all", "full"
        ["q", "v"]
        "q,k,v"
    """
    if mode is None:
        return set()

    if isinstance(mode, str):
        raw = mode.strip().lower()

        if raw in {"", "none", "false", "0", "no", "off"}:
            return set()

        aliases = {
            "all": "qkv",
            "full": "qkv",
            "q_k_v": "qkv",
            "q-k-v": "qkv",
            "q+k+v": "qkv",
            "q,k,v": "qkv",
            "k,v": "kv",
            "q,k": "qk",
            "q,v": "qv",
        }

        raw = aliases.get(raw, raw)

        if "," in raw:
            pieces = [p.strip() for p in raw.split(",") if p.strip()]
        elif "+" in raw:
            pieces = [p.strip() for p in raw.split("+") if p.strip()]
        elif "-" in raw and raw not in {"q-k-v"}:
            pieces = [p.strip() for p in raw.split("-") if p.strip()]
        else:
            pieces = list(raw)

    else:
        pieces = [str(p).strip().lower() for p in mode]

    out: set[str] = set()

    for p in pieces:
        if p in {"query", "queries"}:
            p = "q"
        elif p in {"key", "keys"}:
            p = "k"
        elif p in {"value", "values"}:
            p = "v"

        if p not in {"q", "k", "v"}:
            raise ValueError(
                f"Invalid attention_projection_sharing component {p!r}. "
                "Supported components are q, k, v."
            )

        out.add(p)

    return out


def _validate_projection(
    *,
    name: str,
    proj: nn.Linear | None,
    hidden_size: int,
) -> None:
    if proj is None:
        return

    if not isinstance(proj, nn.Linear):
        raise TypeError(f"{name} must be nn.Linear or None, got {type(proj)!r}.")

    if proj.in_features != hidden_size or proj.out_features != hidden_size:
        raise ValueError(
            f"{name} shape mismatch: expected Linear({hidden_size}, {hidden_size}), "
            f"got Linear({proj.in_features}, {proj.out_features})."
        )


def _maybe_zero_init_linear(layer: nn.Module) -> None:
    if not isinstance(layer, nn.Linear):
        raise RuntimeError(f"Expected nn.Linear, got {type(layer)!r}.")
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _normalize_memory_type(memory_type: str) -> str:
    memory_type = str(memory_type).strip().lower()
    aliases = {
        "none": "identity",
        "skip": "identity",
        "id": "identity",
        "rnn": "gru",
        "gated_recurrent_unit": "gru",
        "ssm": "mamba",
    }
    memory_type = aliases.get(memory_type, memory_type)
    if memory_type not in {"identity", "gru", "mamba"}:
        raise ValueError(
            f"Unknown memory_type={memory_type!r}. "
            "Supported: identity, gru, mamba."
        )
    return memory_type


def _normalize_memory_value_source(value_source: str) -> str:
    value_source = str(value_source).strip().lower()
    aliases = {
        "mem": "memory",
        "historical": "memory",
        "history": "memory",
        "raw": "current",
        "h": "current",
        "x": "current",
        "both": "concat",
        "cat": "concat",
    }
    value_source = aliases.get(value_source, value_source)
    if value_source not in {"memory", "current", "concat"}:
        raise ValueError(
            f"Unknown memory_value_source={value_source!r}. "
            "Supported: memory, current, concat."
        )
    return value_source


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
        emb = self.embedding(pos)

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
        shared_q_proj: nn.Linear | None = None,
        shared_k_proj: nn.Linear | None = None,
        shared_v_proj: nn.Linear | None = None,
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

        _validate_projection(
            name="shared_q_proj",
            proj=shared_q_proj,
            hidden_size=self.hidden_size,
        )
        _validate_projection(
            name="shared_k_proj",
            proj=shared_k_proj,
            hidden_size=self.hidden_size,
        )
        _validate_projection(
            name="shared_v_proj",
            proj=shared_v_proj,
            hidden_size=self.hidden_size,
        )

        self.q_proj = (
            None
            if shared_q_proj is not None
            else nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        )
        self.k_proj = (
            None
            if shared_k_proj is not None
            else nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        )
        self.v_proj = (
            None
            if shared_v_proj is not None
            else nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        )

        object.__setattr__(self, "_shared_q_proj", shared_q_proj)
        object.__setattr__(self, "_shared_k_proj", shared_k_proj)
        object.__setattr__(self, "_shared_v_proj", shared_v_proj)

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

    @property
    def q_linear(self) -> nn.Linear:
        return self._shared_q_proj if self._shared_q_proj is not None else self.q_proj

    @property
    def k_linear(self) -> nn.Linear:
        return self._shared_k_proj if self._shared_k_proj is not None else self.k_proj

    @property
    def v_linear(self) -> nn.Linear:
        return self._shared_v_proj if self._shared_v_proj is not None else self.v_proj

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
        allowed_mask: torch.Tensor | None = None

        if padding_mask is not None:
            key_valid = padding_mask.bool()
            has_any_key = key_valid.any(dim=-1, keepdim=True)
            key_valid_safe = torch.where(has_any_key, key_valid, torch.ones_like(key_valid))
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

        q = self._reshape_heads(self.q_linear(x))
        k = self._reshape_heads(self.k_linear(x))
        v = self._reshape_heads(self.v_linear(x))

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


class _CrossSymbolAttentionBase(nn.Module):
    """
    Shared utilities for symbol-axis attention modules.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        qk_norm: bool,
        attention_gate: bool,
        norm_type: str,
        max_symbols: int,
        use_symbol_embedding: bool,
        symbol_embedding_scale: float,
        mask_self: bool,
        num_register_tokens: int,
        shared_q_proj: nn.Linear | None,
        shared_k_proj: nn.Linear | None,
        shared_v_proj: nn.Linear | None,
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

        _validate_projection(
            name="shared_q_proj",
            proj=shared_q_proj,
            hidden_size=self.hidden_size,
        )
        _validate_projection(
            name="shared_k_proj",
            proj=shared_k_proj,
            hidden_size=self.hidden_size,
        )
        _validate_projection(
            name="shared_v_proj",
            proj=shared_v_proj,
            hidden_size=self.hidden_size,
        )

        self.q_proj = (
            None
            if shared_q_proj is not None
            else nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        )
        self.k_proj = (
            None
            if shared_k_proj is not None
            else nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        )
        self.v_proj = (
            None
            if shared_v_proj is not None
            else nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        )

        object.__setattr__(self, "_shared_q_proj", shared_q_proj)
        object.__setattr__(self, "_shared_k_proj", shared_k_proj)
        object.__setattr__(self, "_shared_v_proj", shared_v_proj)

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

    @property
    def q_linear(self) -> nn.Linear:
        return self._shared_q_proj if self._shared_q_proj is not None else self.q_proj

    @property
    def k_linear(self) -> nn.Linear:
        return self._shared_k_proj if self._shared_k_proj is not None else self.k_proj

    @property
    def v_linear(self) -> nn.Linear:
        return self._shared_v_proj if self._shared_v_proj is not None else self.v_proj

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
        tokens and do not receive ordinary symbol embeddings.
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
                f"Increase max_symbols in cross-symbol config."
            )

        symbol_ids = torch.arange(N_symbols, device=x_btnd.device)
        emb = self.symbol_embedding(symbol_ids)

        out = x_btnd.clone()
        out[:, :, R:, :] = out[:, :, R:, :] + self.symbol_embedding_scale * emb.view(
            1,
            1,
            N_symbols,
            self.hidden_size,
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
            mask_btn = padding_mask.bool().transpose(1, 2).contiguous()

            key_allowed = mask_btn.unsqueeze(2).unsqueeze(3)
            allowed = key_allowed.expand(B, T, 1, N, N)

        if self.mask_self and N > 1:
            not_self = ~torch.eye(N, device=device, dtype=torch.bool).view(1, 1, 1, N, N)
            allowed = not_self if allowed is None else (allowed & not_self)

        if allowed is not None:
            row_has_key = allowed.any(dim=-1, keepdim=True)
            allowed = torch.where(row_has_key, allowed, torch.ones_like(allowed))

        return allowed

    def _symbol_attention(
        self,
        *,
        q_input_btnd: torch.Tensor,
        k_input_btnd: torch.Tensor,
        v_input_btnd: torch.Tensor,
        gate_input_btnd: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        B, T, N, D = q_input_btnd.shape

        if k_input_btnd.shape != (B, T, N, D):
            raise ValueError(
                f"k_input_btnd shape mismatch: expected {(B, T, N, D)}, "
                f"got {tuple(k_input_btnd.shape)}."
            )
        if v_input_btnd.shape != (B, T, N, D):
            raise ValueError(
                f"v_input_btnd shape mismatch: expected {(B, T, N, D)}, "
                f"got {tuple(v_input_btnd.shape)}."
            )

        q = self._reshape_heads(self.q_linear(q_input_btnd))
        k = self._reshape_heads(self.k_linear(k_input_btnd))
        v = self._reshape_heads(self.v_linear(v_input_btnd))

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        attn_mask = self._build_symbol_allowed_mask(
            padding_mask=padding_mask,
            B=B,
            T=T,
            N=N,
            device=q_input_btnd.device,
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
            out = out * torch.sigmoid(self.gate_proj(gate_input_btnd))

        out = self.out_proj(out)
        out = out.transpose(1, 2).contiguous()

        if padding_mask is not None:
            out = out * padding_mask.unsqueeze(-1).to(dtype=out.dtype)

        return out


class CrossSymbolAttention(_CrossSymbolAttentionBase):
    """
    Dense cross-symbol self-attention at each timestamp.

    Input:
        x: (B, N, T, D)

    Output:
        y: (B, N, T, D)
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
        shared_q_proj: nn.Linear | None = None,
        shared_k_proj: nn.Linear | None = None,
        shared_v_proj: nn.Linear | None = None,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            attention_gate=attention_gate,
            norm_type=norm_type,
            max_symbols=max_symbols,
            use_symbol_embedding=use_symbol_embedding,
            symbol_embedding_scale=symbol_embedding_scale,
            mask_self=mask_self,
            num_register_tokens=num_register_tokens,
            shared_q_proj=shared_q_proj,
            shared_k_proj=shared_k_proj,
            shared_v_proj=shared_v_proj,
        )

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

        x_btnd = x.transpose(1, 2).contiguous()
        x_btnd = self._add_symbol_embedding(x_btnd)

        return self._symbol_attention(
            q_input_btnd=x_btnd,
            k_input_btnd=x_btnd,
            v_input_btnd=x_btnd,
            gate_input_btnd=x_btnd,
            padding_mask=padding_mask,
        )

class CrossSymbolLaggedAttention(_CrossSymbolAttentionBase):
    """
    Cross-symbol lagged attention.

    For each query h_i(t), attend to lagged states:

        { h_j(t - lag) : j = 1..N, lag in lag_set }

    Input:
        x: (B, N, T, D)

    Output:
        y: (B, N, T, D)

    Notes:
        - This module is causal as long as all lags are >= 0.
        - K/V length is N * len(lag_set).
        - lag=0 is allowed by default, but self lag0 can be masked.
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
        mask_self_lag0: bool = True,
        mask_self_all_lags: bool = False,
        num_register_tokens: int = 0,
        lag_set: Sequence[int] = (0, 1, 2, 3, 5, 8, 13, 20, 30),
        use_lag_embedding: bool = True,
        lag_embedding_scale: float = 1.0,
        use_lag_bias: bool = True,
        per_head_lag_bias: bool = True,
        shared_q_proj: nn.Linear | None = None,
        shared_k_proj: nn.Linear | None = None,
        shared_v_proj: nn.Linear | None = None,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            attention_gate=attention_gate,
            norm_type=norm_type,
            max_symbols=max_symbols,
            use_symbol_embedding=use_symbol_embedding,
            symbol_embedding_scale=symbol_embedding_scale,
            mask_self=False,  # handled manually because key axis is N * L
            num_register_tokens=num_register_tokens,
            shared_q_proj=shared_q_proj,
            shared_k_proj=shared_k_proj,
            shared_v_proj=shared_v_proj,
        )

        lags = sorted({int(lag) for lag in lag_set})
        if not lags:
            raise ValueError("lag_set cannot be empty.")
        if min(lags) < 0:
            raise ValueError(f"All lags must be non-negative, got {lags}.")

        self.lag_set = tuple(lags)
        self.num_lags = len(self.lag_set)
        self.max_lag = max(self.lag_set)

        self.mask_self_lag0 = bool(mask_self_lag0)
        self.mask_self_all_lags = bool(mask_self_all_lags)

        self.use_lag_embedding = bool(use_lag_embedding)
        self.lag_embedding_scale = float(lag_embedding_scale)
        self.use_lag_bias = bool(use_lag_bias)
        self.per_head_lag_bias = bool(per_head_lag_bias)

        if self.use_lag_embedding:
            self.lag_embedding = nn.Embedding(self.num_lags, self.hidden_size)
        else:
            self.lag_embedding = None

        if self.use_lag_bias:
            if self.per_head_lag_bias:
                self.lag_bias = nn.Parameter(torch.zeros(self.num_heads, self.num_lags))
            else:
                self.lag_bias = nn.Parameter(torch.zeros(1, self.num_lags))
        else:
            self.lag_bias = None

    def _shift_time(
        self,
        x_btn_or_btnd: torch.Tensor,
        lag: int,
        *,
        fill_value: float | bool = 0,
    ) -> torch.Tensor:
        """
        Return y[t] = x[t - lag], with invalid early positions filled.

        Supports:
            x: (B, T, N)
            x: (B, T, N, D)
        """
        if lag == 0:
            return x_btn_or_btnd

        out = torch.empty_like(x_btn_or_btnd)

        if x_btn_or_btnd.dtype == torch.bool:
            out.fill_(bool(fill_value))
        else:
            out.fill_(float(fill_value))

        out[:, lag:, ...] = x_btn_or_btnd[:, :-lag, ...]
        return out

    def _make_lagged_bank(
        self,
        x_btnd: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x_btnd:
            (B, T, N, D)

        padding_mask:
            (B, N, T), optional

        Returns:
            lagged_x:
                (B, T, N, L, D)

            key_valid:
                (B, T, N, L), bool
        """
        B, T, N, D = x_btnd.shape

        lagged = []
        valid = []

        if padding_mask is None:
            mask_btn = torch.ones((B, T, N), device=x_btnd.device, dtype=torch.bool)
        else:
            mask_btn = padding_mask.bool().transpose(1, 2).contiguous()

        for lag in self.lag_set:
            lagged.append(self._shift_time(x_btnd, lag, fill_value=0.0))
            valid.append(self._shift_time(mask_btn, lag, fill_value=False))

        lagged_x = torch.stack(lagged, dim=3)  # (B, T, N, L, D)
        key_valid = torch.stack(valid, dim=3)  # (B, T, N, L)

        if self.lag_embedding is not None:
            lag_ids = torch.arange(self.num_lags, device=x_btnd.device)
            lag_emb = self.lag_embedding(lag_ids).to(dtype=lagged_x.dtype)
            lagged_x = lagged_x + self.lag_embedding_scale * lag_emb.view(
                1, 1, 1, self.num_lags, self.hidden_size
            )

        return lagged_x, key_valid

    def _build_lagged_allowed_mask(
        self,
        *,
        key_valid_btnl: torch.Tensor,
        B: int,
        T: int,
        N: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Return bool mask broadcastable to:

            (B, T, H, N_query, N_key * L)

        True means allowed.
        """
        L = self.num_lags

        allowed = key_valid_btnl.reshape(B, T, 1, 1, N * L)
        allowed = allowed.expand(B, T, 1, N, N * L)

        if self.mask_self_lag0 or self.mask_self_all_lags:
            q_symbol = torch.arange(N, device=device).view(N, 1, 1)
            k_symbol = torch.arange(N, device=device).view(1, N, 1)
            lag_ids = torch.arange(L, device=device).view(1, 1, L)

            same_symbol = q_symbol == k_symbol

            if self.mask_self_all_lags:
                disallow = same_symbol.expand(N, N, L)
            else:
                lag0_index = self.lag_set.index(0) if 0 in self.lag_set else None
                if lag0_index is None:
                    disallow = torch.zeros((N, N, L), device=device, dtype=torch.bool)
                else:
                    disallow = same_symbol & (lag_ids == lag0_index)

            allow_pair_lag = ~disallow.reshape(N, N * L)
            allowed = allowed & allow_pair_lag.view(1, 1, 1, N, N * L)

        # Avoid all-masked rows causing NaNs in SDPA.
        row_has_key = allowed.any(dim=-1, keepdim=True)
        allowed = torch.where(row_has_key, allowed, torch.ones_like(allowed))

        return allowed

    def _build_lag_bias(
        self,
        *,
        B: int,
        T: int,
        N: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        """
        Return additive bias:

            (1, 1, H, 1, N * L)
        """
        if self.lag_bias is None:
            return None

        L = self.num_lags

        if self.per_head_lag_bias:
            bias_hl = self.lag_bias.to(dtype=dtype, device=device)  # (H, L)
        else:
            bias_hl = self.lag_bias.to(dtype=dtype, device=device).expand(
                self.num_heads, L
            )

        # Repeat lag bias for each symbol.
        # key order is: (symbol, lag), flattened from (N, L).
        bias_hnl = bias_hl.view(self.num_heads, 1, L).expand(self.num_heads, N, L)
        bias_hk = bias_hnl.reshape(self.num_heads, N * L)

        return bias_hk.view(1, 1, self.num_heads, 1, N * L)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"CrossSymbolLaggedAttention expects x with shape (B,N,T,D), "
                f"got {tuple(x.shape)}."
            )

        B, N, T, D = x.shape
        if D != self.hidden_size:
            raise ValueError(f"Expected hidden_size={self.hidden_size}, got {D}.")

        if padding_mask is not None and padding_mask.shape != (B, N, T):
            raise ValueError(
                f"padding_mask shape mismatch: expected {(B, N, T)}, "
                f"got {tuple(padding_mask.shape)}."
            )

        x_btnd = x.transpose(1, 2).contiguous()  # (B, T, N, D)

        # Add symbol embedding before building lagged bank, so each lagged key
        # still carries symbol identity.
        x_btnd = self._add_symbol_embedding(x_btnd)

        q_input = x_btnd
        lagged_x, key_valid = self._make_lagged_bank(
            x_btnd=x_btnd,
            padding_mask=padding_mask,
        )

        # Flatten key axis from (N, L) to (N * L).
        k_input = lagged_x.reshape(B, T, N * self.num_lags, D)
        v_input = k_input

        q = self._reshape_heads(self.q_linear(q_input))  # (B, T, H, N, Hd)
        k = self._reshape_heads(self.k_linear(k_input))  # (B, T, H, N*L, Hd)
        v = self._reshape_heads(self.v_linear(v_input))  # (B, T, H, N*L, Hd)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        allowed = self._build_lagged_allowed_mask(
            key_valid_btnl=key_valid,
            B=B,
            T=T,
            N=N,
            device=x.device,
        )

        lag_bias = self._build_lag_bias(
            B=B,
            T=T,
            N=N,
            dtype=q.dtype,
            device=x.device,
        )

        if lag_bias is None:
            attn_mask = allowed
        else:
            neg_inf = torch.finfo(q.dtype).min
            attn_mask = lag_bias.masked_fill(~allowed, neg_inf)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
            scale=self.scale,
        )

        out = self._merge_heads(out)  # (B, T, N, D)

        if self.gate_proj is not None:
            out = out * torch.sigmoid(self.gate_proj(q_input))

        out = self.out_proj(out)
        out = out.transpose(1, 2).contiguous()  # (B, N, T, D)

        if padding_mask is not None:
            out = out * padding_mask.unsqueeze(-1).to(dtype=out.dtype)

        return out

class CausalSymbolMemory(nn.Module):
    """
    Per-symbol causal temporal memory.

    Input:
        x: (B, N, T, D)

    Output:
        memory: (B, N, T, D)

    Modes:
        identity:
            memory = x

        gru:
            memory_core = GRU over T for each (B, N) independently.

        mamba:
            memory_core = mamba_ssm.Mamba over T for each (B, N) independently.

    Default is residual memory with zero-init output projection:
        memory = x + memory_residual_scale * up(core(down(norm(x))))

    With memory_output_zero_init=True, memory starts exactly as x.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        memory_type: str = "gru",
        memory_hidden_size: int | None = None,
        memory_num_layers: int = 1,
        memory_dropout: float = 0.0,
        memory_residual: bool = True,
        memory_residual_scale: float = 1.0,
        memory_output_zero_init: bool = True,
        norm_type: str = "rmsnorm",
        norm_eps: float = 1e-6,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
    ) -> None:
        super().__init__()

        self.hidden_size = int(hidden_size)
        self.memory_type = _normalize_memory_type(memory_type)
        self.memory_hidden_size = int(memory_hidden_size or hidden_size)
        self.memory_num_layers = int(memory_num_layers)
        self.memory_dropout_p = float(memory_dropout)
        self.memory_residual = bool(memory_residual)
        self.memory_residual_scale = float(memory_residual_scale)
        self.memory_output_zero_init = bool(memory_output_zero_init)

        self.norm = build_norm(norm_type, self.hidden_size, eps=norm_eps)

        if self.memory_type == "identity":
            self.down_proj = nn.Identity()
            self.core = nn.Identity()
            self.up_proj = nn.Identity()
            self.dropout = nn.Dropout(self.memory_dropout_p)
            return

        self.down_proj = (
            nn.Identity()
            if self.memory_hidden_size == self.hidden_size
            else nn.Linear(self.hidden_size, self.memory_hidden_size)
        )

        if self.memory_type == "gru":
            if self.memory_num_layers <= 0:
                raise ValueError(
                    f"memory_num_layers must be positive for GRU, got {self.memory_num_layers}"
                )

            self.core = nn.GRU(
                input_size=self.memory_hidden_size,
                hidden_size=self.memory_hidden_size,
                num_layers=self.memory_num_layers,
                batch_first=True,
                dropout=self.memory_dropout_p if self.memory_num_layers > 1 else 0.0,
            )

        elif self.memory_type == "mamba":
            try:
                from mamba_ssm import Mamba
            except Exception as exc:
                raise ImportError(
                    "memory_type='mamba' requires the optional package `mamba-ssm`. "
                    "Install it in your environment, or use memory_type='gru'."
                ) from exc

            self.core = Mamba(
                d_model=self.memory_hidden_size,
                d_state=int(mamba_d_state),
                d_conv=int(mamba_d_conv),
                expand=int(mamba_expand),
            )

        else:
            raise RuntimeError(f"Unexpected memory_type={self.memory_type!r}")

        self.up_proj = (
            nn.Identity()
            if self.memory_hidden_size == self.hidden_size
            else nn.Linear(self.memory_hidden_size, self.hidden_size)
        )

        self.dropout = nn.Dropout(self.memory_dropout_p)

        if self.memory_output_zero_init:
            if isinstance(self.up_proj, nn.Linear):
                _maybe_zero_init_linear(self.up_proj)

    def _run_core(self, x_flat: torch.Tensor) -> torch.Tensor:
        """
        x_flat:
            (B*N, T, D_mem)
        """
        if self.memory_type == "identity":
            return x_flat

        if self.memory_type == "gru":
            out, _ = self.core(x_flat)
            return out

        if self.memory_type == "mamba":
            return self.core(x_flat)

        raise RuntimeError(f"Unexpected memory_type={self.memory_type!r}")

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"CausalSymbolMemory expects x with shape (B,N,T,D), got {tuple(x.shape)}."
            )

        B, N, T, D = x.shape
        if D != self.hidden_size:
            raise ValueError(f"Expected hidden_size={self.hidden_size}, got {D}.")

        if padding_mask is not None and padding_mask.shape != (B, N, T):
            raise ValueError(
                f"padding_mask shape mismatch: expected {(B, N, T)}, "
                f"got {tuple(padding_mask.shape)}."
            )

        if self.memory_type == "identity":
            out = x
            if padding_mask is not None:
                out = out * padding_mask.unsqueeze(-1).to(dtype=out.dtype)
            return out

        y = self.norm(x)

        if padding_mask is not None:
            y = y * padding_mask.unsqueeze(-1).to(dtype=y.dtype)

        y = self.down_proj(y)

        D_mem = y.shape[-1]
        y_flat = y.reshape(B * N, T, D_mem)

        if padding_mask is not None:
            mask_flat = padding_mask.reshape(B * N, T).unsqueeze(-1).to(dtype=y_flat.dtype)
            y_flat = y_flat * mask_flat

        out_flat = self._run_core(y_flat)

        if padding_mask is not None:
            out_flat = out_flat * mask_flat

        out = out_flat.reshape(B, N, T, D_mem)
        out = self.dropout(out)
        out = self.up_proj(out)

        if padding_mask is not None:
            out = out * padding_mask.unsqueeze(-1).to(dtype=out.dtype)

        if self.memory_residual:
            out = x + self.memory_residual_scale * out

        if padding_mask is not None:
            out = out * padding_mask.unsqueeze(-1).to(dtype=out.dtype)

        return out


class CrossSymbolMemoryAttention(_CrossSymbolAttentionBase):
    """
    Cross-symbol attention where:

        Q comes from current h_i(t)
        K comes from causal memory m_j(t)
        V comes from one of:
            - memory:  m_j(t)
            - current: h_j(t)
            - concat:  W[h_j(t), m_j(t)]

    This is the "current state queries other symbols' historical summary" layer.

    Input:
        x: (B, N, T, D)

    Output:
        y: (B, N, T, D)
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
        norm_eps: float = 1e-6,
        max_symbols: int = 512,
        use_symbol_embedding: bool = True,
        symbol_embedding_scale: float = 1.0,
        mask_self: bool = False,
        num_register_tokens: int = 0,
        memory_type: str = "gru",
        memory_hidden_size: int | None = None,
        memory_num_layers: int = 1,
        memory_dropout: float = 0.0,
        memory_residual: bool = True,
        memory_residual_scale: float = 1.0,
        memory_output_zero_init: bool = True,
        memory_value_source: str = "memory",
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        shared_q_proj: nn.Linear | None = None,
        shared_k_proj: nn.Linear | None = None,
        shared_v_proj: nn.Linear | None = None,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            attention_gate=attention_gate,
            norm_type=norm_type,
            max_symbols=max_symbols,
            use_symbol_embedding=use_symbol_embedding,
            symbol_embedding_scale=symbol_embedding_scale,
            mask_self=mask_self,
            num_register_tokens=num_register_tokens,
            shared_q_proj=shared_q_proj,
            shared_k_proj=shared_k_proj,
            shared_v_proj=shared_v_proj,
        )

        self.memory_value_source = _normalize_memory_value_source(memory_value_source)

        self.memory = CausalSymbolMemory(
            hidden_size=hidden_size,
            memory_type=memory_type,
            memory_hidden_size=memory_hidden_size,
            memory_num_layers=memory_num_layers,
            memory_dropout=memory_dropout,
            memory_residual=memory_residual,
            memory_residual_scale=memory_residual_scale,
            memory_output_zero_init=memory_output_zero_init,
            norm_type=norm_type,
            norm_eps=norm_eps,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
        )

        self.concat_value_proj = (
            nn.Linear(2 * hidden_size, hidden_size)
            if self.memory_value_source == "concat"
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"CrossSymbolMemoryAttention expects x with shape (B,N,T,D), "
                f"got {tuple(x.shape)}."
            )

        B, N, T, D = x.shape
        if D != self.hidden_size:
            raise ValueError(f"Expected hidden_size={self.hidden_size}, got {D}.")

        if padding_mask is not None and padding_mask.shape != (B, N, T):
            raise ValueError(
                f"padding_mask shape mismatch: expected {(B, N, T)}, "
                f"got {tuple(padding_mask.shape)}."
            )

        memory = self.memory(x, padding_mask=padding_mask)

        x_btnd = x.transpose(1, 2).contiguous()
        memory_btnd = memory.transpose(1, 2).contiguous()

        q_input = self._add_symbol_embedding(x_btnd)
        k_input = self._add_symbol_embedding(memory_btnd)

        if self.memory_value_source == "memory":
            v_input = k_input
        elif self.memory_value_source == "current":
            v_input = q_input
        elif self.memory_value_source == "concat":
            if self.concat_value_proj is None:
                raise RuntimeError("concat_value_proj is None for memory_value_source='concat'.")
            v_input = self.concat_value_proj(torch.cat([q_input, k_input], dim=-1))
        else:
            raise RuntimeError(f"Unexpected memory_value_source={self.memory_value_source!r}")

        return self._symbol_attention(
            q_input_btnd=q_input,
            k_input_btnd=k_input,
            v_input_btnd=v_input,
            gate_input_btnd=q_input,
            padding_mask=padding_mask,
        )


class CrossSymbolFiLM(nn.Module):
    """
    Use dense cross-symbol attention as a conditioning branch.
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
        norm_eps: float = 1e-6,
        max_symbols: int = 512,
        use_symbol_embedding: bool = True,
        symbol_embedding_scale: float = 1.0,
        mask_self: bool = False,
        num_register_tokens: int = 0,
        film_hidden_size: int | None = None,
        use_beta: bool = True,
        gamma_activation: str = "tanh",
        zero_init: bool = True,
        shared_q_proj: nn.Linear | None = None,
        shared_k_proj: nn.Linear | None = None,
        shared_v_proj: nn.Linear | None = None,
    ) -> None:
        super().__init__()

        self.hidden_size = int(hidden_size)
        self.use_beta = bool(use_beta)
        self.gamma_activation = str(gamma_activation).strip().lower()

        self.cross = CrossSymbolAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            attention_gate=attention_gate,
            norm_type=norm_type,
            max_symbols=max_symbols,
            use_symbol_embedding=use_symbol_embedding,
            symbol_embedding_scale=symbol_embedding_scale,
            mask_self=mask_self,
            num_register_tokens=num_register_tokens,
            shared_q_proj=shared_q_proj,
            shared_k_proj=shared_k_proj,
            shared_v_proj=shared_v_proj,
        )

        out_dim = 2 * hidden_size if self.use_beta else hidden_size

        self.cond_norm = build_norm(norm_type, hidden_size, eps=norm_eps)

        if film_hidden_size is None:
            self.film_mlp = nn.Linear(hidden_size, out_dim)
            final_layer = self.film_mlp
        else:
            inner = int(film_hidden_size)
            self.film_mlp = nn.Sequential(
                nn.Linear(hidden_size, inner),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(inner, out_dim),
            )
            final_layer = self.film_mlp[-1]

        if zero_init:
            _maybe_zero_init_linear(final_layer)

    def _activate_gamma(self, gamma: torch.Tensor) -> torch.Tensor:
        if self.gamma_activation == "identity":
            return gamma
        if self.gamma_activation == "tanh":
            return torch.tanh(gamma)
        if self.gamma_activation == "sigmoid":
            return 2.0 * torch.sigmoid(gamma) - 1.0
        raise KeyError(f"Unknown gamma_activation={self.gamma_activation!r}")

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(
                f"CrossSymbolFiLM expects x with shape (B,N,T,D), got {tuple(x.shape)}."
            )

        cond = self.cross(x, padding_mask=padding_mask)
        cond = self.cond_norm(cond)

        params = self.film_mlp(cond)

        if self.use_beta:
            gamma, beta = params.chunk(2, dim=-1)
        else:
            gamma = params
            beta = torch.zeros_like(gamma)

        gamma = self._activate_gamma(gamma)

        if padding_mask is not None:
            valid = padding_mask.unsqueeze(-1).to(dtype=gamma.dtype)
            gamma = gamma * valid
            beta = beta * valid

        return gamma, beta


class CrossSymbolMemoryFiLM(nn.Module):
    """
    Use memory cross-symbol attention as a conditioning branch.

    Q from current h, K/V from causal memory, then:
        gamma, beta = film_mlp(context)
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
        norm_eps: float = 1e-6,
        max_symbols: int = 512,
        use_symbol_embedding: bool = True,
        symbol_embedding_scale: float = 1.0,
        mask_self: bool = False,
        num_register_tokens: int = 0,
        film_hidden_size: int | None = None,
        use_beta: bool = True,
        gamma_activation: str = "tanh",
        zero_init: bool = True,
        memory_type: str = "gru",
        memory_hidden_size: int | None = None,
        memory_num_layers: int = 1,
        memory_dropout: float = 0.0,
        memory_residual: bool = True,
        memory_residual_scale: float = 1.0,
        memory_output_zero_init: bool = True,
        memory_value_source: str = "memory",
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        shared_q_proj: nn.Linear | None = None,
        shared_k_proj: nn.Linear | None = None,
        shared_v_proj: nn.Linear | None = None,
    ) -> None:
        super().__init__()

        self.hidden_size = int(hidden_size)
        self.use_beta = bool(use_beta)
        self.gamma_activation = str(gamma_activation).strip().lower()

        self.cross = CrossSymbolMemoryAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            attention_gate=attention_gate,
            norm_type=norm_type,
            norm_eps=norm_eps,
            max_symbols=max_symbols,
            use_symbol_embedding=use_symbol_embedding,
            symbol_embedding_scale=symbol_embedding_scale,
            mask_self=mask_self,
            num_register_tokens=num_register_tokens,
            memory_type=memory_type,
            memory_hidden_size=memory_hidden_size,
            memory_num_layers=memory_num_layers,
            memory_dropout=memory_dropout,
            memory_residual=memory_residual,
            memory_residual_scale=memory_residual_scale,
            memory_output_zero_init=memory_output_zero_init,
            memory_value_source=memory_value_source,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            shared_q_proj=shared_q_proj,
            shared_k_proj=shared_k_proj,
            shared_v_proj=shared_v_proj,
        )

        out_dim = 2 * hidden_size if self.use_beta else hidden_size

        self.cond_norm = build_norm(norm_type, hidden_size, eps=norm_eps)

        if film_hidden_size is None:
            self.film_mlp = nn.Linear(hidden_size, out_dim)
            final_layer = self.film_mlp
        else:
            inner = int(film_hidden_size)
            self.film_mlp = nn.Sequential(
                nn.Linear(hidden_size, inner),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(inner, out_dim),
            )
            final_layer = self.film_mlp[-1]

        if zero_init:
            _maybe_zero_init_linear(final_layer)

    def _activate_gamma(self, gamma: torch.Tensor) -> torch.Tensor:
        if self.gamma_activation == "identity":
            return gamma
        if self.gamma_activation == "tanh":
            return torch.tanh(gamma)
        if self.gamma_activation == "sigmoid":
            return 2.0 * torch.sigmoid(gamma) - 1.0
        raise KeyError(f"Unknown gamma_activation={self.gamma_activation!r}")

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(
                f"CrossSymbolMemoryFiLM expects x with shape (B,N,T,D), got {tuple(x.shape)}."
            )

        cond = self.cross(x, padding_mask=padding_mask)
        cond = self.cond_norm(cond)

        params = self.film_mlp(cond)

        if self.use_beta:
            gamma, beta = params.chunk(2, dim=-1)
        else:
            gamma = params
            beta = torch.zeros_like(gamma)

        gamma = self._activate_gamma(gamma)

        if padding_mask is not None:
            valid = padding_mask.unsqueeze(-1).to(dtype=gamma.dtype)
            gamma = gamma * valid
            beta = beta * valid

        return gamma, beta

class TemporalGRU(nn.Module):
    """
    Causal temporal GRU sublayer.

    Supports:
        x: (B, T, D)
        x: (B, N, T, D)

    For (B, N, T, D), it runs one independent GRU sequence per (B, N).
    No cross-symbol mixing happens here.

    This module returns a residual branch output with the same shape as input.
    `_ResidualSubLayer` is responsible for:
        x <- x + residual_scale * dropout(out)

    With output_zero_init=True, this branch starts as an exact zero update.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        gru_hidden_size: int | None = None,
        gru_num_layers: int = 1,
        gru_dropout: float = 0.0,
        bias: bool = True,
        output_zero_init: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_size = int(hidden_size)
        self.gru_hidden_size = int(gru_hidden_size or hidden_size)
        self.gru_num_layers = int(gru_num_layers)

        if self.gru_num_layers <= 0:
            raise ValueError(
                f"gru_num_layers must be positive, got {self.gru_num_layers}"
            )

        self.in_proj = (
            nn.Identity()
            if self.gru_hidden_size == self.hidden_size
            else nn.Linear(self.hidden_size, self.gru_hidden_size, bias=bias)
        )

        self.gru = nn.GRU(
            input_size=self.gru_hidden_size,
            hidden_size=self.gru_hidden_size,
            num_layers=self.gru_num_layers,
            batch_first=True,
            dropout=float(gru_dropout) if self.gru_num_layers > 1 else 0.0,
            bias=bias,
            bidirectional=False,
        )

        # Always use an explicit output projection so we can zero-init the
        # residual branch even when gru_hidden_size == hidden_size.
        self.out_proj = nn.Linear(self.gru_hidden_size, self.hidden_size, bias=bias)

        if output_zero_init:
            nn.init.zeros_(self.out_proj.weight)
            if self.out_proj.bias is not None:
                nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x:
            (B, T, D)
            or
            (B, N, T, D)

        padding_mask:
            (B, T)
            or
            (B, N, T)
        """
        if x.ndim not in (3, 4):
            raise ValueError(
                f"TemporalGRU expects x with shape (B,T,D) or (B,N,T,D), "
                f"got {tuple(x.shape)}."
            )

        if padding_mask is not None and padding_mask.shape != x.shape[:-1]:
            raise ValueError(
                f"padding_mask shape mismatch: expected {tuple(x.shape[:-1])}, "
                f"got {tuple(padding_mask.shape)}."
            )

        *leading_shape, T, D = x.shape

        if D != self.hidden_size:
            raise ValueError(f"Expected hidden_size={self.hidden_size}, got {D}.")

        x_flat = x.reshape(-1, T, D)

        mask_flat: torch.Tensor | None = None
        if padding_mask is not None:
            mask_flat = padding_mask.reshape(-1, T).unsqueeze(-1).to(dtype=x.dtype)
            x_flat = x_flat * mask_flat

        z = self.in_proj(x_flat)
        out, _ = self.gru(z)
        out = self.out_proj(out)

        if mask_flat is not None:
            out = out * mask_flat

        return out.reshape(*leading_shape, T, D)

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

            "cross_symbol_lagged_attention": "cross_symbol_lagged_attention",
            "cross-symbol-lagged-attention": "cross_symbol_lagged_attention",
            "cross_lagged_attention": "cross_symbol_lagged_attention",
            "cross-lagged-attention": "cross_symbol_lagged_attention",
            "lagged_cross_attention": "cross_symbol_lagged_attention",
            "lagged_cross_symbol_attention": "cross_symbol_lagged_attention",
            "lead_lag_attention": "cross_symbol_lagged_attention",
            "lead-lag-attention": "cross_symbol_lagged_attention",
            
            "cross_symbol_film": "cross_symbol_film",
            "cross-symbol-film": "cross_symbol_film",
            "symbol_film": "cross_symbol_film",
            "symbol-film": "cross_symbol_film",
            "cross_film": "cross_symbol_film",
            "cross-film": "cross_symbol_film",

            "cross_symbol_memory_attention": "cross_symbol_memory_attention",
            "cross-symbol-memory-attention": "cross_symbol_memory_attention",
            "cross_memory_attention": "cross_symbol_memory_attention",
            "cross-memory-attention": "cross_symbol_memory_attention",
            "symbol_memory_attention": "cross_symbol_memory_attention",
            "memory_cross_attention": "cross_symbol_memory_attention",
            "memory_cross_symbol_attention": "cross_symbol_memory_attention",

            "cross_symbol_memory_film": "cross_symbol_memory_film",
            "cross-symbol-memory-film": "cross_symbol_memory_film",
            "cross_memory_film": "cross_symbol_memory_film",
            "cross-memory-film": "cross_symbol_memory_film",
            "symbol_memory_film": "cross_symbol_memory_film",
            "memory_cross_film": "cross_symbol_memory_film",
            "memory_cross_symbol_film": "cross_symbol_memory_film",
            
            "gru": "temporal_gru",
            "rnn": "temporal_gru",
            "temporal_gru": "temporal_gru",
            "temporal-gru": "temporal_gru",
            "recurrent": "temporal_gru",
            "temporal_recurrent": "temporal_gru",
            "temporal-recurrent": "temporal_gru",
            
            "mlp": "ffn",
            "feedforward": "ffn",
            "feed_forward": "ffn",
            "dense_ffn": "ffn",

            "moe": "moe_ffn",
            "moeffn": "moe_ffn",
            "moe-feedforward": "moe_ffn",
            "moe_feedforward": "moe_ffn",
            "codebook": "codebook_adapter",
            "codebook_adapter": "codebook_adapter",
        }

        layer_type = aliases.get(layer_type, layer_type)

        if layer_type not in {
            "attention",
            "cross_symbol_attention",
            "cross_symbol_film",
            "cross_symbol_memory_attention",
            "cross_symbol_memory_film",
            "temporal_gru",
            "ffn",
            "moe_ffn",
            "cross_symbol_lagged_attention",
            "codebook_adapter",
        }:
            raise ValueError(
                f"Unknown sublayer type {layer_type!r}. "
                "Supported: attention, cross_symbol_attention, cross_symbol_film, "
                "cross_symbol_memory_attention, cross_symbol_memory_film, "
                "cross_symbol_lagged_attention, "
                "temporal_gru, ffn, moe_ffn, codebook_adapter."
            )

        spec["type"] = layer_type
        out.append(spec)

    if not out:
        raise ValueError("sublayers cannot be empty.")

    return out


class _ResidualSubLayer(nn.Module):
    """
    One PreNorm sublayer.

    Supports:
        - attention
        - cross_symbol_attention
        - cross_symbol_film
        - cross_symbol_memory_attention
        - cross_symbol_memory_film
        - ffn
        - moe_ffn
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
        film_mode: str = "residual_norm",
    ) -> None:
        super().__init__()

        self.layer_type = str(layer_type)
        self.module = module
        self.norm = build_norm(norm_type, hidden_size, eps=norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = float(residual_scale)
        self.film_mode = str(film_mode).strip().lower()

    def _apply_film(
        self,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        if self.film_mode == "residual_norm":
            out = gamma * y + beta
            return x + self.residual_scale * self.dropout(out)

        if self.film_mode == "direct":
            gamma = self.dropout(gamma)
            beta = self.dropout(beta)
            return x * (1.0 + self.residual_scale * gamma) + self.residual_scale * beta

        raise KeyError(
            f"Unknown film_mode={self.film_mode!r}. "
            "Supported: residual_norm, direct."
        )

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
            x = x + self.residual_scale * self.dropout(out)

        elif self.layer_type == "temporal_gru":
            out = self.module(y, padding_mask=padding_mask)
            x = x + self.residual_scale * self.dropout(out)

        elif self.layer_type == "cross_symbol_attention":
            out = self.module(y, padding_mask=padding_mask)
            x = x + self.residual_scale * self.dropout(out)
            
        elif self.layer_type == "cross_symbol_lagged_attention":
            out = self.module(y, padding_mask=padding_mask)
            x = x + self.residual_scale * self.dropout(out)
            
        elif self.layer_type == "cross_symbol_memory_attention":
            out = self.module(y, padding_mask=padding_mask)
            x = x + self.residual_scale * self.dropout(out)

        elif self.layer_type == "cross_symbol_film":
            gamma, beta = self.module(y, padding_mask=padding_mask)
            x = self._apply_film(x=x, y=y, gamma=gamma, beta=beta)

        elif self.layer_type == "cross_symbol_memory_film":
            gamma, beta = self.module(y, padding_mask=padding_mask)
            x = self._apply_film(x=x, y=y, gamma=gamma, beta=beta)

        elif self.layer_type == "ffn":
            out = self.module(y, padding_mask=padding_mask)
            x = x + self.residual_scale * self.dropout(out)

        elif self.layer_type == "moe_ffn":
            out, losses, metrics = self.module(y, padding_mask=padding_mask)
            x = x + self.residual_scale * self.dropout(out)
        elif self.layer_type == "codebook_adapter":
            out, aux = self.module(x, return_aux=True)
            x = out
            metrics["codebook_entropy"] = float(aux["code_attn_entropy"].detach().item())
            metrics["codebook_effective_num"] = float(aux["code_effective_num"].detach().item())
            metrics["codebook_gate"] = float(aux["code_gate"].detach().item())

        else:
            raise RuntimeError(f"Unexpected layer_type={self.layer_type!r}")

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

        attention_projection_sharing: str | Sequence[str] | None = "none",
        codebook_enabled: bool = False,
        codebook_num_codes: int = 128,
        codebook_num_heads: int = 4,
        codebook_dropout: float = 0.0,
        codebook_topk: int | None = None,
        codebook_temperature: float = 1.0,
        codebook_residual_gate_init: float = 0.0,
        codebook_use_layernorm: bool = True,
        codebook_share_kv: bool = False,
        codebook_position: str = "after_ffn",
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

        self.attention_projection_sharing = _normalize_attention_projection_sharing(
            attention_projection_sharing
        )

        self.shared_q_proj = (
            nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            if "q" in self.attention_projection_sharing
            else None
        )
        self.shared_k_proj = (
            nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            if "k" in self.attention_projection_sharing
            else None
        )
        self.shared_v_proj = (
            nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            if "v" in self.attention_projection_sharing
            else None
        )

        dense_ffn_hidden_size = int(ffn_hidden_size or 4 * hidden_size)

        specs = _normalize_sublayer_specs(
            sublayers,
            use_moe=self.use_moe,
        )
        if codebook_enabled:
            codebook_position = str(codebook_position).strip().lower()
            if codebook_position not in {"after_attn", "after_ffn"}:
                raise ValueError("codebook_position must be one of: after_attn, after_ffn.")
            codebook_spec = {
                "type": "codebook_adapter",
                "num_codes": int(codebook_num_codes),
                "num_heads": int(codebook_num_heads),
                "dropout": float(codebook_dropout),
                "topk": codebook_topk,
                "temperature": float(codebook_temperature),
                "residual_gate_init": float(codebook_residual_gate_init),
                "use_layernorm": bool(codebook_use_layernorm),
                "share_kv_codebook": bool(codebook_share_kv),
            }
            insert_idx = len(specs)
            if codebook_position == "after_attn":
                for j, spec in enumerate(specs):
                    if str(spec.get("type", "")).strip().lower() == "attention":
                        insert_idx = j + 1
                        break
            specs.insert(insert_idx, codebook_spec)

        layers: list[_ResidualSubLayer] = []

        for i, spec in enumerate(specs):
            spec = dict(spec)
            layer_type = str(spec.pop("type")).strip().lower()

            layer_dropout = float(spec.pop("dropout", dropout))
            layer_residual_scale = float(spec.pop("residual_scale", residual_scale))
            layer_norm_type = str(spec.pop("norm_type", norm_type))
            layer_norm_eps = float(spec.pop("norm_eps", norm_eps))
            layer_film_mode = "residual_norm"

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
                    shared_q_proj=self.shared_q_proj,
                    shared_k_proj=self.shared_k_proj,
                    shared_v_proj=self.shared_v_proj,
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
                    shared_q_proj=self.shared_q_proj,
                    shared_k_proj=self.shared_k_proj,
                    shared_v_proj=self.shared_v_proj,
                )
                
            elif layer_type == "cross_symbol_lagged_attention":
                module = CrossSymbolLaggedAttention(
                    hidden_size=self.hidden_size,
                    num_heads=int(spec.pop("num_heads", num_heads)),
                    dropout=layer_dropout,
                    qk_norm=bool(spec.pop("qk_norm", qk_norm)),
                    attention_gate=bool(spec.pop("attention_gate", attention_gate)),
                    norm_type=str(spec.pop("qk_norm_type", norm_type)),
                    max_symbols=int(spec.pop("max_symbols", 512)),
                    use_symbol_embedding=bool(spec.pop("use_symbol_embedding", True)),
                    symbol_embedding_scale=float(spec.pop("symbol_embedding_scale", 1.0)),
                    mask_self_lag0=bool(spec.pop("mask_self_lag0", True)),
                    mask_self_all_lags=bool(spec.pop("mask_self_all_lags", False)),
                    num_register_tokens=int(
                        spec.pop("num_register_tokens", self.num_register_tokens)
                    ),
                    lag_set=spec.pop("lag_set", (0, 1, 2, 3, 5, 8, 13, 20, 30)),
                    use_lag_embedding=bool(spec.pop("use_lag_embedding", True)),
                    lag_embedding_scale=float(spec.pop("lag_embedding_scale", 1.0)),
                    use_lag_bias=bool(spec.pop("use_lag_bias", True)),
                    per_head_lag_bias=bool(spec.pop("per_head_lag_bias", True)),
                    shared_q_proj=self.shared_q_proj,
                    shared_k_proj=self.shared_k_proj,
                    shared_v_proj=self.shared_v_proj,
                )
                
            elif layer_type == "cross_symbol_memory_attention":
                module = CrossSymbolMemoryAttention(
                    hidden_size=self.hidden_size,
                    num_heads=int(spec.pop("num_heads", num_heads)),
                    dropout=layer_dropout,
                    qk_norm=bool(spec.pop("qk_norm", qk_norm)),
                    attention_gate=bool(spec.pop("attention_gate", attention_gate)),
                    norm_type=str(spec.pop("qk_norm_type", norm_type)),
                    norm_eps=float(spec.pop("memory_norm_eps", layer_norm_eps)),
                    max_symbols=int(spec.pop("max_symbols", 512)),
                    use_symbol_embedding=bool(spec.pop("use_symbol_embedding", True)),
                    symbol_embedding_scale=float(spec.pop("symbol_embedding_scale", 1.0)),
                    mask_self=bool(spec.pop("mask_self", False)),
                    num_register_tokens=int(
                        spec.pop("num_register_tokens", self.num_register_tokens)
                    ),
                    memory_type=str(spec.pop("memory_type", "gru")),
                    memory_hidden_size=spec.pop("memory_hidden_size", None),
                    memory_num_layers=int(spec.pop("memory_num_layers", 1)),
                    memory_dropout=float(spec.pop("memory_dropout", 0.0)),
                    memory_residual=bool(spec.pop("memory_residual", True)),
                    memory_residual_scale=float(spec.pop("memory_residual_scale", 1.0)),
                    memory_output_zero_init=bool(
                        spec.pop("memory_output_zero_init", True)
                    ),
                    memory_value_source=str(spec.pop("memory_value_source", "memory")),
                    mamba_d_state=int(spec.pop("mamba_d_state", 16)),
                    mamba_d_conv=int(spec.pop("mamba_d_conv", 4)),
                    mamba_expand=int(spec.pop("mamba_expand", 2)),
                    shared_q_proj=self.shared_q_proj,
                    shared_k_proj=self.shared_k_proj,
                    shared_v_proj=self.shared_v_proj,
                )

            elif layer_type == "cross_symbol_film":
                layer_film_mode = str(spec.pop("film_mode", "residual_norm"))

                film_norm_type = str(spec.pop("film_norm_type", norm_type))
                film_norm_eps = float(spec.pop("film_norm_eps", layer_norm_eps))

                module = CrossSymbolFiLM(
                    hidden_size=self.hidden_size,
                    num_heads=int(spec.pop("num_heads", num_heads)),
                    dropout=layer_dropout,
                    qk_norm=bool(spec.pop("qk_norm", qk_norm)),
                    attention_gate=bool(spec.pop("attention_gate", attention_gate)),
                    norm_type=str(spec.pop("qk_norm_type", film_norm_type)),
                    norm_eps=film_norm_eps,
                    max_symbols=int(spec.pop("max_symbols", 512)),
                    use_symbol_embedding=bool(spec.pop("use_symbol_embedding", True)),
                    symbol_embedding_scale=float(spec.pop("symbol_embedding_scale", 1.0)),
                    mask_self=bool(spec.pop("mask_self", False)),
                    num_register_tokens=int(
                        spec.pop("num_register_tokens", self.num_register_tokens)
                    ),
                    film_hidden_size=spec.pop("film_hidden_size", None),
                    use_beta=bool(spec.pop("use_beta", True)),
                    gamma_activation=str(spec.pop("gamma_activation", "tanh")),
                    zero_init=bool(spec.pop("zero_init", True)),
                    shared_q_proj=self.shared_q_proj,
                    shared_k_proj=self.shared_k_proj,
                    shared_v_proj=self.shared_v_proj,
                )

            elif layer_type == "cross_symbol_memory_film":
                layer_film_mode = str(spec.pop("film_mode", "residual_norm"))

                film_norm_type = str(spec.pop("film_norm_type", norm_type))
                film_norm_eps = float(spec.pop("film_norm_eps", layer_norm_eps))

                module = CrossSymbolMemoryFiLM(
                    hidden_size=self.hidden_size,
                    num_heads=int(spec.pop("num_heads", num_heads)),
                    dropout=layer_dropout,
                    qk_norm=bool(spec.pop("qk_norm", qk_norm)),
                    attention_gate=bool(spec.pop("attention_gate", attention_gate)),
                    norm_type=str(spec.pop("qk_norm_type", film_norm_type)),
                    norm_eps=film_norm_eps,
                    max_symbols=int(spec.pop("max_symbols", 512)),
                    use_symbol_embedding=bool(spec.pop("use_symbol_embedding", True)),
                    symbol_embedding_scale=float(spec.pop("symbol_embedding_scale", 1.0)),
                    mask_self=bool(spec.pop("mask_self", False)),
                    num_register_tokens=int(
                        spec.pop("num_register_tokens", self.num_register_tokens)
                    ),
                    film_hidden_size=spec.pop("film_hidden_size", None),
                    use_beta=bool(spec.pop("use_beta", True)),
                    gamma_activation=str(spec.pop("gamma_activation", "tanh")),
                    zero_init=bool(spec.pop("zero_init", True)),
                    memory_type=str(spec.pop("memory_type", "gru")),
                    memory_hidden_size=spec.pop("memory_hidden_size", None),
                    memory_num_layers=int(spec.pop("memory_num_layers", 1)),
                    memory_dropout=float(spec.pop("memory_dropout", 0.0)),
                    memory_residual=bool(spec.pop("memory_residual", True)),
                    memory_residual_scale=float(spec.pop("memory_residual_scale", 1.0)),
                    memory_output_zero_init=bool(
                        spec.pop("memory_output_zero_init", True)
                    ),
                    memory_value_source=str(spec.pop("memory_value_source", "memory")),
                    mamba_d_state=int(spec.pop("mamba_d_state", 16)),
                    mamba_d_conv=int(spec.pop("mamba_d_conv", 4)),
                    mamba_expand=int(spec.pop("mamba_expand", 2)),
                    shared_q_proj=self.shared_q_proj,
                    shared_k_proj=self.shared_k_proj,
                    shared_v_proj=self.shared_v_proj,
                )
                
            elif layer_type == "temporal_gru":
                module = TemporalGRU(
                    hidden_size=self.hidden_size,
                    gru_hidden_size=spec.pop("gru_hidden_size", None),
                    gru_num_layers=int(spec.pop("gru_num_layers", 1)),
                    gru_dropout=float(spec.pop("gru_dropout", 0.0)),
                    bias=bool(spec.pop("bias", True)),
                    output_zero_init=bool(spec.pop("output_zero_init", True)),
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
            elif layer_type == "codebook_adapter":
                module = CodebookAdapter(
                    d_model=self.hidden_size,
                    num_codes=int(spec.pop("num_codes", 128)),
                    num_heads=int(spec.pop("num_heads", 4)),
                    dropout=float(spec.pop("dropout", 0.0)),
                    topk=spec.pop("topk", None),
                    temperature=float(spec.pop("temperature", 1.0)),
                    residual_gate_init=float(spec.pop("residual_gate_init", 0.0)),
                    use_layernorm=bool(spec.pop("use_layernorm", True)),
                    share_kv_codebook=bool(spec.pop("share_kv_codebook", False)),
                    return_aux=bool(spec.pop("return_aux", False)),
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
                    film_mode=layer_film_mode,
                )
            )

        self.layers = nn.ModuleList(layers)

    def _prepend_register_tokens(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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
        if not self.use_register_tokens or self.return_register_tokens:
            return x

        return x[:, self.num_register_tokens :, :, :].contiguous()

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
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

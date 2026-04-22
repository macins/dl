from __future__ import annotations

from .norms import build_norm
from .registry import register_block
from ...utils.externals import ensure_torch

torch = ensure_torch()
nn = torch.nn
F = nn.functional


def _build_position_ids(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def _apply_rope(x: torch.Tensor, position_ids: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    head_dim = x.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires even head_dim, got {head_dim}")
    freq_seq = torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32)
    inv_freq = 1.0 / (float(base) ** (freq_seq / head_dim))
    angles = position_ids.to(dtype=torch.float32).unsqueeze(-1) * inv_freq.view(1, 1, -1)
    cos = torch.repeat_interleave(torch.cos(angles), repeats=2, dim=-1).unsqueeze(1)
    sin = torch.repeat_interleave(torch.sin(angles), repeats=2, dim=-1).unsqueeze(1)
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
    distance = (positions.view(1, 1, seq_len, 1) - positions.view(1, 1, 1, seq_len)).abs()
    bias = -slopes * distance
    return bias.to(dtype=dtype)


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
        if self.mode != "absolute":
            return x
        batch_size, seq_len, _ = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}")
        pos = _build_position_ids(batch_size, seq_len, x.device)
        return x + self.embedding(pos)

    def apply_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode != "rope":
            return q, k
        batch_size, _, seq_len, _ = q.shape
        pos = _build_position_ids(batch_size, seq_len, q.device)
        return _apply_rope(q, pos), _apply_rope(k, pos)

    def attention_bias(self, x: torch.Tensor) -> torch.Tensor | None:
        if self.mode != "alibi":
            return None
        _, _, seq_len = x.shape[:3]
        return _build_alibi_bias(
            num_heads=1,
            seq_len=seq_len,
            device=x.device,
            dtype=x.dtype,
        )


class MultiHeadSelfAttention(nn.Module):
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
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qk_norm = bool(qk_norm)
        self.use_attention_gate = bool(attention_gate)

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.q_norm = build_norm(norm_type, self.head_dim) if self.qk_norm else nn.Identity()
        self.k_norm = build_norm(norm_type, self.head_dim) if self.qk_norm else nn.Identity()
        self.gate_proj = nn.Linear(self.hidden_size, self.hidden_size) if self.use_attention_gate else None
        self.position = PositionEmbedding(
            model_dim=self.hidden_size,
            max_seq_len=max_seq_len,
            mode=position_encoding,
        )

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self._reshape_heads(self.q_proj(x))
        k = self._reshape_heads(self.k_proj(x))
        v = self._reshape_heads(self.v_proj(x))

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k = self.position.apply_qk(q, k)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if self.position.mode == "alibi":
            alibi = _build_alibi_bias(
                num_heads=self.num_heads,
                seq_len=x.shape[1],
                device=x.device,
                dtype=scores.dtype,
            )
            scores = scores + alibi

        if padding_mask is not None:
            key_mask = padding_mask.bool().unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)
        if padding_mask is not None:
            query_mask = padding_mask.bool().unsqueeze(1).unsqueeze(-1)
            attn = attn * query_mask.to(dtype=attn.dtype)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = self._merge_heads(out)

        if self.gate_proj is not None:
            out = out * torch.sigmoid(self.gate_proj(x))

        out = self.out_proj(out)
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
        self.ffn_hidden_size = int(ffn_hidden_size or 4 * hidden_size)
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

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        out = self._activate(self.in_proj(x))
        out = self.out_proj(self.dropout(out))
        if padding_mask is not None:
            out = out * padding_mask.unsqueeze(-1).to(dtype=out.dtype)
        return out


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
    ) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.attn_norm = build_norm(norm_type, hidden_size, eps=norm_eps)
        self.ffn_norm = build_norm(norm_type, hidden_size, eps=norm_eps)
        self.attn = MultiHeadSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            attention_gate=attention_gate,
            norm_type=norm_type,
            max_seq_len=max_seq_len,
            position_encoding=position_encoding,
        )
        self.ffn = FeedForward(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            activation=ffn_activation,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out = self.attn(self.attn_norm(x), padding_mask=padding_mask)
        x = x + self.residual_scale * self.dropout(attn_out)
        ffn_out = self.ffn(self.ffn_norm(x), padding_mask=padding_mask)
        x = x + self.residual_scale * self.dropout(ffn_out)
        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1).to(dtype=x.dtype)
        return x

from __future__ import annotations

from collections.abc import Sequence

from ..base import BaseModel
from ..registry import register_model
from ..layers.transformer import FeedForward, MoEFeedForward
from ...utils.externals import ensure_torch

torch = ensure_torch()
nn = torch.nn


class _CausalSelfAttnBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, ffn_activation: str = "swiglu",
                 use_moe: bool = False, num_experts: int = 8, top_k: int = 2, shared_experts: int = 0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = (
            MoEFeedForward(
                hidden_size=d_model,
                dense_ffn_hidden_size=d_ff,
                num_experts=num_experts,
                top_k=top_k,
                shared_experts=shared_experts,
                activation=ffn_activation,
                dropout=dropout,
            )
            if use_moe
            else FeedForward(hidden_size=d_model, ffn_hidden_size=d_ff, activation=ffn_activation, dropout=dropout)
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        y = self.ln1(x)
        y, _ = self.attn(y, y, y, attn_mask=attn_mask, need_weights=False)
        x = x + y
        x = x + self.ffn(self.ln2(x))
        return x


class _CrossAttnBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, ffn_activation: str = "swiglu",
                 use_moe: bool = False, num_experts: int = 8, top_k: int = 2, shared_experts: int = 0) -> None:
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.cross = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = (
            MoEFeedForward(
                hidden_size=d_model,
                dense_ffn_hidden_size=d_ff,
                num_experts=num_experts,
                top_k=top_k,
                shared_experts=shared_experts,
                activation=ffn_activation,
                dropout=dropout,
            )
            if use_moe
            else FeedForward(hidden_size=d_model, ffn_hidden_size=d_ff, activation=ffn_activation, dropout=dropout)
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        y, _ = self.cross(self.ln_q(q), self.ln_kv(kv), self.ln_kv(kv), attn_mask=attn_mask, need_weights=False)
        q = q + y
        q = q + self.ffn(self.ln2(q))
        return q


@register_model("encoder_decoder_sequence_regressor")
class EncoderDecoderSequenceRegressor(BaseModel):
    def __init__(self, *, input_dim: int, d_model: int = 256, d_ff: int = 512, n_heads: int = 8, dropout: float = 0.1,
                 adapter_type: str = "linear", adapter_norm: bool = True,
                 n_local_layers: int = 2, use_product_memory: bool = True, n_product_layers: int = 1,
                 use_market_memory: bool = True, n_market_layers: int = 1,
                 use_factor_memory: bool = True, num_factor_tokens: int = 16, n_factor_temporal_layers: int = 1,
                 factor_cross_n_heads: int = 4, n_decoder_layers: int = 2,
                 decoder_memory_order: Sequence[str] = ("local", "product", "market", "factor"),
                 num_horizons: int = 1, use_symbol_embedding: bool = False, num_symbols: int | None = None,
                 use_time_embedding: bool = False, max_time_steps: int | None = None,
                 use_horizon_embedding: bool = True, use_product_embedding: bool = False,
                 output_mode: str = "single_horizon", head_type: str = "linear", head_hidden_dim: int = 256,
                 ffn_activation: str = "swiglu", use_moe: bool = True, num_experts: int = 8, top_k: int = 2,
                 shared_experts: int = 0,
                 target_key: str = "ret_30min") -> None:
        super().__init__()
        if adapter_type != "linear":
            raise ValueError("only adapter_type=linear is supported")
        self.target_key = target_key
        self.use_product_memory = bool(use_product_memory)
        self.use_market_memory = bool(use_market_memory)
        self.use_factor_memory = bool(use_factor_memory)
        self.use_symbol_embedding = bool(use_symbol_embedding)
        self.use_time_embedding = bool(use_time_embedding)
        self.use_horizon_embedding = bool(use_horizon_embedding)
        self.use_product_embedding = bool(use_product_embedding)
        self.output_mode = str(output_mode)
        self.num_horizons = int(num_horizons)
        self.decoder_memory_order = [str(m) for m in decoder_memory_order]

        self.adapter = nn.Linear(input_dim, d_model)
        self.adapter_norm = nn.LayerNorm(d_model) if adapter_norm else nn.Identity()
        self.adapter_drop = nn.Dropout(dropout)

        self.local_blocks = nn.ModuleList([_CausalSelfAttnBlock(d_model, n_heads, d_ff, dropout, ffn_activation, use_moe, num_experts, top_k, shared_experts) for _ in range(n_local_layers)])
        self.product_blocks = nn.ModuleList([_CausalSelfAttnBlock(d_model, n_heads, d_ff, dropout, ffn_activation, use_moe, num_experts, top_k, shared_experts) for _ in range(n_product_layers)])
        self.market_blocks = nn.ModuleList([_CausalSelfAttnBlock(d_model, n_heads, d_ff, dropout, ffn_activation, use_moe, num_experts, top_k, shared_experts) for _ in range(n_market_layers)])
        self.factor_temporal_blocks = nn.ModuleList([_CausalSelfAttnBlock(d_model, n_heads, d_ff, dropout, ffn_activation, use_moe, num_experts, top_k, shared_experts) for _ in range(n_factor_temporal_layers)])

        self.factor_query = nn.Parameter(torch.randn(num_factor_tokens, d_model) * 0.02)
        self.factor_cross = nn.MultiheadAttention(d_model, factor_cross_n_heads, dropout=dropout, batch_first=True)

        self.query_proj = nn.Linear(d_model, d_model)
        self.symbol_embedding = nn.Embedding(num_symbols, d_model) if self.use_symbol_embedding else None
        self.time_embedding = nn.Embedding(max_time_steps, d_model) if self.use_time_embedding else None
        self.horizon_embedding = nn.Embedding(self.num_horizons, d_model) if self.use_horizon_embedding else None
        self.product_embedding = nn.Embedding(4096, d_model) if self.use_product_embedding else None

        self.decoder_layers = nn.ModuleList([nn.ModuleDict({
            "local": _CrossAttnBlock(d_model, n_heads, d_ff, dropout, ffn_activation, use_moe, num_experts, top_k, shared_experts),
            "product": _CrossAttnBlock(d_model, n_heads, d_ff, dropout, ffn_activation, use_moe, num_experts, top_k, shared_experts),
            "market": _CrossAttnBlock(d_model, n_heads, d_ff, dropout, ffn_activation, use_moe, num_experts, top_k, shared_experts),
            "factor": _CrossAttnBlock(d_model, n_heads, d_ff, dropout, ffn_activation, use_moe, num_experts, top_k, shared_experts),
        }) for _ in range(n_decoder_layers)])

        self.head_norm = nn.LayerNorm(d_model)
        if head_type == "linear":
            self.head = nn.Linear(d_model, 1)
        else:
            self.head = nn.Sequential(nn.Linear(d_model, head_hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(head_hidden_dim, 1))

    def _causal_mask(self, tq: int, tk: int, device: torch.device) -> torch.Tensor:
        q_idx = torch.arange(tq, device=device).unsqueeze(1)
        k_idx = torch.arange(tk, device=device).unsqueeze(0)
        return k_idx > q_idx

    def _factor_mask(self, t: int, k: int, device: torch.device) -> torch.Tensor:
        q_idx = torch.arange(t, device=device).unsqueeze(1)
        mem_t = torch.arange(t, device=device).repeat_interleave(k).unsqueeze(0)
        return mem_t > q_idx

    def _encode_temporal(self, x: torch.Tensor, blocks: nn.ModuleList) -> torch.Tensor:
        if not blocks:
            return x
        mask = self._causal_mask(x.shape[1], x.shape[1], x.device)
        for blk in blocks:
            x = blk(x, mask)
        return x

    def forward(self, batch: dict) -> dict:
        x = batch["features"]["continuous"]
        b, n, t, _ = x.shape
        product_ids = batch.get("product_ids")
        if product_ids is None:
            product_ids = torch.arange(n, device=x.device).unsqueeze(0).expand(b, n)
        elif product_ids.ndim == 1:
            product_ids = product_ids.unsqueeze(0).expand(b, -1)

        h = self.adapter_drop(self.adapter_norm(self.adapter(x)))
        l = self._encode_temporal(h.reshape(b * n, t, -1), self.local_blocks).reshape(b, n, t, -1)

        p_mem = None
        if self.use_product_memory:
            num_p = int(product_ids.max().item()) + 1
            one_hot = torch.nn.functional.one_hot(product_ids, num_classes=num_p).float()
            denom = one_hot.sum(dim=1).clamp_min(1.0).unsqueeze(-1).unsqueeze(-1)
            p_mem = torch.einsum("bnp,bntd->bptd", one_hot, l) / denom
            p_mem = self._encode_temporal(p_mem.reshape(b * num_p, t, -1), self.product_blocks).reshape(b, num_p, t, -1)

        g_mem = None
        if self.use_market_memory:
            g_mem = l.mean(dim=1)
            g_mem = self._encode_temporal(g_mem, self.market_blocks)

        z_mem = None
        if self.use_factor_memory:
            kv = l.permute(0, 2, 1, 3).reshape(b * t, n, -1)
            qf = self.factor_query.unsqueeze(0).expand(b * t, -1, -1)
            z, _ = self.factor_cross(qf, kv, kv, need_weights=False)
            k = z.shape[1]
            z_mem = z.reshape(b, t, k, -1).permute(0, 2, 1, 3)
            z_mem = self._encode_temporal(z_mem.reshape(b * k, t, -1), self.factor_temporal_blocks).reshape(b, k, t, -1)

        q = self.query_proj(l).unsqueeze(3).expand(b, n, t, self.num_horizons, -1)
        if self.symbol_embedding is not None:
            sid = batch.get("symbol_ids")
            if sid is None:
                sid = torch.arange(n, device=x.device)
            if sid.ndim == 1:
                sid = sid.unsqueeze(0).expand(b, -1)
            q = q + self.symbol_embedding(sid).unsqueeze(2).unsqueeze(3)
        if self.time_embedding is not None:
            tids = batch.get("time_ids", torch.arange(t, device=x.device))
            if tids.ndim == 1:
                tids = tids.unsqueeze(0).expand(b, -1)
            q = q + self.time_embedding(tids).unsqueeze(1).unsqueeze(3)
        if self.horizon_embedding is not None:
            q = q + self.horizon_embedding(torch.arange(self.num_horizons, device=x.device)).view(1, 1, 1, self.num_horizons, -1)
        if self.product_embedding is not None:
            q = q + self.product_embedding(product_ids).unsqueeze(2).unsqueeze(3)

        q_flat = q.permute(0, 1, 3, 2, 4).reshape(b * n * self.num_horizons, t, -1)
        causal_tt = self._causal_mask(t, t, x.device)
        for layer in self.decoder_layers:
            for mem_name in self.decoder_memory_order:
                if mem_name == "local":
                    kv = l.unsqueeze(2).expand(b, n, self.num_horizons, t, l.shape[-1]).reshape(b * n * self.num_horizons, t, -1)
                    q_flat = layer["local"](q_flat, kv, causal_tt)
                elif mem_name == "product" and p_mem is not None:
                    p_for_symbol = p_mem.gather(1, product_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, t, p_mem.shape[-1]))
                    kv = p_for_symbol.unsqueeze(2).expand(b, n, self.num_horizons, t, p_mem.shape[-1]).reshape(b * n * self.num_horizons, t, -1)
                    q_flat = layer["product"](q_flat, kv, causal_tt)
                elif mem_name == "market" and g_mem is not None:
                    kv = g_mem.unsqueeze(1).unsqueeze(2).expand(b, n, self.num_horizons, t, g_mem.shape[-1]).reshape(b * n * self.num_horizons, t, -1)
                    q_flat = layer["market"](q_flat, kv, causal_tt)
                elif mem_name == "factor" and z_mem is not None:
                    kf = z_mem.shape[1]
                    kv = z_mem.permute(0, 2, 1, 3).reshape(b, t * kf, -1).unsqueeze(1).unsqueeze(2).expand(b, n, self.num_horizons, t * kf, z_mem.shape[-1]).reshape(b * n * self.num_horizons, t * kf, -1)
                    q_flat = layer["factor"](q_flat, kv, self._factor_mask(t, kf, x.device))

        q_out = q_flat.reshape(b, n, self.num_horizons, t, -1).permute(0, 1, 3, 2, 4)
        pred = self.head(self.head_norm(q_out)).squeeze(-1)
        if self.output_mode == "single_horizon":
            pred = pred[..., 0]
        return {"preds": {self.target_key: pred}}

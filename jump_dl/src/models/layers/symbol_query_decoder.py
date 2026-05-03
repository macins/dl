from __future__ import annotations

from typing import Any

import torch
from torch import nn


class SymbolQueryDecoder(nn.Module):
    """Decoder-stage cross-symbol readout over panel hidden states.

    Input/Output shape: [B, N, T, D].

    Modes:
    - full_causal: full cross-symbol causal memory over all (symbol, time<=t).
      Complexity is O((N*T)^2), which can be expensive for large N/T.
    - topk_static: each target symbol only reads from configured source symbols.
    - product_memory: market-memory mode using pooled symbol memory over time.
    - self_plus_cross: combines own-history and cross-symbol contexts with a gate.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mode: str = "full_causal",
        dropout: float = 0.0,
        use_symbol_embedding: bool = True,
        num_symbols: int | None = None,
        residual_init: float = 0.0,
        use_layer_norm: bool = True,
        exclude_self: bool = False,
        add_lag_bias: bool = False,
        max_lag: int | None = None,
        topk_indices: torch.Tensor | None = None,
        topk_k: int | None = None,
        memory_pooling: str = "mean",
        cross_gate: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.mode = mode
        self.use_symbol_embedding = use_symbol_embedding
        self.num_symbols = num_symbols
        self.exclude_self = exclude_self
        self.add_lag_bias = add_lag_bias
        self.max_lag = max_lag
        self.topk_k = topk_k
        self.memory_pooling = memory_pooling
        self.cross_gate = cross_gate

        self.pre_norm = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        self.q_proj = nn.Linear(d_model, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.residual_alpha = nn.Parameter(torch.tensor(float(residual_init)))

        self.symbol_embedding = None
        if self.use_symbol_embedding and self.num_symbols is not None:
            self.symbol_embedding = nn.Embedding(self.num_symbols, d_model)

        self.register_buffer("_topk_indices", topk_indices.long() if topk_indices is not None else None, persistent=False)

        # TODO: lag bias is intentionally deferred to keep a minimal first implementation.

        self.gate_mlp = None
        if self.mode == "self_plus_cross" and self.cross_gate:
            self.gate_mlp = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )

    def _resolve_symbol_ids(self, h: torch.Tensor, symbol_ids: torch.Tensor | None) -> torch.Tensor | None:
        if not self.use_symbol_embedding:
            return None
        bsz, n_sym, _, _ = h.shape
        if symbol_ids is not None:
            if symbol_ids.dim() == 1:
                if symbol_ids.shape[0] != n_sym:
                    raise ValueError(f"symbol_ids [N] mismatch: got {symbol_ids.shape[0]} vs N={n_sym}")
                return symbol_ids.to(h.device).view(1, n_sym).expand(bsz, -1)
            if symbol_ids.dim() == 2:
                if tuple(symbol_ids.shape) != (bsz, n_sym):
                    raise ValueError(f"symbol_ids [B,N] mismatch: got {tuple(symbol_ids.shape)} vs {(bsz, n_sym)}")
                return symbol_ids.to(h.device)
            raise ValueError("symbol_ids must have shape [N] or [B, N]")

        if self.num_symbols is not None:
            if n_sym > self.num_symbols:
                raise ValueError(f"N={n_sym} exceeds configured num_symbols={self.num_symbols}")
            return torch.arange(n_sym, device=h.device).view(1, n_sym).expand(bsz, -1)

        raise ValueError("use_symbol_embedding=True requires symbol_ids or num_symbols")

    def _time_causal_mask(self, tgt_t: int, src_t: int, device: torch.device) -> torch.Tensor:
        t = torch.arange(tgt_t, device=device).view(-1, 1)
        s = torch.arange(src_t, device=device).view(1, -1)
        return s > t

    def _full_causal_mask(self, n_sym: int, t_steps: int, device: torch.device) -> torch.Tensor:
        ti = torch.arange(t_steps, device=device).repeat(n_sym)
        tj = torch.arange(t_steps, device=device).repeat(n_sym)
        mask = tj.view(1, -1) > ti.view(-1, 1)
        if self.exclude_self:
            si = torch.arange(n_sym, device=device).repeat_interleave(t_steps)
            sj = torch.arange(n_sym, device=device).repeat_interleave(t_steps)
            mask = mask | (sj.view(1, -1) == si.view(-1, 1))
        return mask

    def _apply_symbol_embeddings(self, q: torch.Tensor, ids_bn: torch.Tensor | None) -> torch.Tensor:
        if ids_bn is None:
            return q
        if self.symbol_embedding is None:
            self.symbol_embedding = nn.Embedding(self.num_symbols or int(ids_bn.max().item() + 1), self.d_model).to(q.device)
        return q + self.symbol_embedding(ids_bn).unsqueeze(2)

    def forward(
        self,
        h: torch.Tensor,
        symbol_ids: torch.Tensor | None = None,
        topk_indices: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_mask
        bsz, n_sym, t_steps, d_model = h.shape
        if d_model != self.d_model:
            raise ValueError(f"d_model mismatch: got {d_model}, expected {self.d_model}")

        q = self.q_proj(self.pre_norm(h))
        ids_bn = self._resolve_symbol_ids(h, symbol_ids)
        q = self._apply_symbol_embeddings(q, ids_bn)

        if self.mode == "full_causal":
            q_flat = q.reshape(bsz, n_sym * t_steps, d_model)
            kv_flat = h.reshape(bsz, n_sym * t_steps, d_model)
            attn_mask = self._full_causal_mask(n_sym, t_steps, h.device)
            decoded, _ = self.attn(q_flat, kv_flat, kv_flat, attn_mask=attn_mask, need_weights=False)
            decoded = decoded.reshape(bsz, n_sym, t_steps, d_model)

        elif self.mode == "product_memory":
            if self.memory_pooling != "mean":
                raise ValueError(f"Unsupported memory_pooling: {self.memory_pooling}")
            mem = h.mean(dim=1)
            q_bn = q.reshape(bsz * n_sym, t_steps, d_model)
            mem_bn = mem.unsqueeze(1).expand(-1, n_sym, -1, -1).reshape(bsz * n_sym, t_steps, d_model)
            attn_mask = self._time_causal_mask(t_steps, t_steps, h.device)
            decoded, _ = self.attn(q_bn, mem_bn, mem_bn, attn_mask=attn_mask, need_weights=False)
            decoded = decoded.reshape(bsz, n_sym, t_steps, d_model)

        elif self.mode == "topk_static":
            tk = topk_indices if topk_indices is not None else self._topk_indices
            if tk is None:
                raise ValueError("topk_static mode requires topk_indices in constructor or forward")
            if tk.shape[0] != n_sym:
                raise ValueError(f"topk_indices first dim must be N={n_sym}, got {tk.shape[0]}")
            tk = tk.to(h.device).long()
            outs = []
            causal_tk = self._time_causal_mask(t_steps, tk.shape[1] * t_steps, h.device)
            for i in range(n_sym):
                q_i = q[:, i]
                kv_i = h[:, tk[i]].reshape(bsz, tk.shape[1] * t_steps, d_model)
                out_i, _ = self.attn(q_i, kv_i, kv_i, attn_mask=causal_tk, need_weights=False)
                outs.append(out_i)
            decoded = torch.stack(outs, dim=1)

        elif self.mode == "self_plus_cross":
            own = []
            cross = []
            own_mask = self._time_causal_mask(t_steps, t_steps, h.device)
            cross_mask = self._full_causal_mask(n_sym, t_steps, h.device)
            q_flat = q.reshape(bsz, n_sym * t_steps, d_model)
            kv_flat = h.reshape(bsz, n_sym * t_steps, d_model)
            cross_flat, _ = self.attn(q_flat, kv_flat, kv_flat, attn_mask=cross_mask, need_weights=False)
            cross_all = cross_flat.reshape(bsz, n_sym, t_steps, d_model)
            for i in range(n_sym):
                own_i, _ = self.attn(q[:, i], h[:, i], h[:, i], attn_mask=own_mask, need_weights=False)
                own.append(own_i)
                cross.append(cross_all[:, i])
            own_ctx = torch.stack(own, dim=1)
            cross_ctx = torch.stack(cross, dim=1)
            if self.gate_mlp is not None:
                gate = torch.sigmoid(self.gate_mlp(h))
                decoded = gate * own_ctx + (1.0 - gate) * cross_ctx
            else:
                decoded = 0.5 * (own_ctx + cross_ctx)
        else:
            raise ValueError(f"Unknown SymbolQueryDecoder mode: {self.mode}")

        return h + self.residual_alpha * self.dropout(decoded)

    def get_aux_stats(self) -> dict[str, Any]:
        return {
            "symbol_query/residual_alpha": float(self.residual_alpha.detach().cpu().item()),
            "symbol_query/mode": self.mode,
            "symbol_query/topk_k": self.topk_k,
        }

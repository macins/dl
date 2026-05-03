from __future__ import annotations

from typing import Any

import torch
from torch import nn


class PersistentMemoryBank(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_market_slots: int = 16,
        num_symbol_slots: int = 0,
        num_symbols: int | None = None,
        memory_levels: list[str] | tuple[str, ...] = ("market",),
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_market_slots = int(num_market_slots)
        self.num_symbol_slots = int(num_symbol_slots)
        self.memory_levels = set(memory_levels)
        self.num_symbols_total = num_symbols

        self.market_memory = None
        self.symbol_memory = None

        if "market" in self.memory_levels:
            self.market_memory = nn.Parameter(torch.empty(self.num_market_slots, self.d_model))
            nn.init.normal_(self.market_memory, mean=0.0, std=init_std)

        if "symbol" in self.memory_levels:
            if num_symbols is None:
                raise ValueError("num_symbols must be provided when symbol memory is enabled")
            if self.num_symbol_slots <= 0:
                raise ValueError("num_symbol_slots must be > 0 when symbol memory is enabled")
            self.symbol_memory = nn.Parameter(torch.empty(int(num_symbols), self.num_symbol_slots, self.d_model))
            nn.init.normal_(self.symbol_memory, mean=0.0, std=init_std)

    def get_memory(self, batch_size: int, num_symbols: int, symbol_ids: torch.Tensor | None = None, device=None, dtype=None) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        if self.market_memory is not None:
            market = self.market_memory.to(device=device, dtype=dtype)
            chunks.append(market.view(1, 1, self.num_market_slots, self.d_model).expand(batch_size, num_symbols, -1, -1))

        if self.symbol_memory is not None:
            sym = self.symbol_memory.to(device=device, dtype=dtype)
            if symbol_ids is None:
                if sym.size(0) < num_symbols:
                    raise ValueError(f"Configured symbol memory has {sym.size(0)} symbols but got N={num_symbols}")
                ids = torch.arange(num_symbols, device=sym.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
            else:
                if symbol_ids.dim() == 1:
                    ids = symbol_ids.long().unsqueeze(0).expand(batch_size, -1)
                elif symbol_ids.dim() == 2:
                    ids = symbol_ids.long()
                else:
                    raise ValueError(f"symbol_ids must be [N] or [B,N], got {tuple(symbol_ids.shape)}")
                ids = ids.to(sym.device)
            if ids.size(1) != num_symbols:
                raise ValueError(f"symbol_ids second dim must be N={num_symbols}, got {ids.size(1)}")
            sym_tokens = sym[ids]  # [B,N,K,D]
            chunks.append(sym_tokens)

        if not chunks:
            raise ValueError("No memory levels enabled in PersistentMemoryBank")
        return torch.cat(chunks, dim=2)


class PrecomputedMemoryEncoder(nn.Module):
    def __init__(self, summary_dim: int, d_model: int, num_summary_slots: int = 4, encoder_type: str = "mlp", pooling: str = "mean", include_market_summary: bool = True, include_symbol_summary: bool = True) -> None:
        super().__init__()
        self.summary_dim = int(summary_dim)
        self.d_model = int(d_model)
        self.num_summary_slots = int(num_summary_slots)
        self.pooling = pooling
        self.include_market_summary = include_market_summary
        self.include_symbol_summary = include_symbol_summary
        if encoder_type != "mlp":
            raise ValueError(f"Unsupported encoder_type={encoder_type}")
        self.encoder = nn.Sequential(nn.Linear(self.summary_dim, self.d_model), nn.GELU(), nn.Linear(self.d_model, self.d_model))

    def forward(self, rolling_memory: torch.Tensor, num_symbols: int) -> torch.Tensor:
        if rolling_memory.dim() == 4 and rolling_memory.size(-1) == self.d_model:
            return rolling_memory
        if rolling_memory.dim() == 3 and rolling_memory.size(-1) == self.d_model:
            return rolling_memory.unsqueeze(1).expand(-1, num_symbols, -1, -1)
        if rolling_memory.dim() == 4 and rolling_memory.size(-1) == self.summary_dim:
            b, n, l, _ = rolling_memory.shape
            enc = self.encoder(rolling_memory)
            if self.pooling == "none" or self.num_summary_slots == l:
                return enc
            pooled = enc.mean(dim=2, keepdim=True)
            return pooled.expand(b, n, self.num_summary_slots, self.d_model)
        if rolling_memory.dim() == 3 and rolling_memory.size(-1) == self.summary_dim:
            b, l, _ = rolling_memory.shape
            enc = self.encoder(rolling_memory)
            if self.pooling != "none" and self.num_summary_slots != l:
                enc = enc.mean(dim=1, keepdim=True).expand(b, self.num_summary_slots, self.d_model)
            return enc.unsqueeze(1).expand(-1, num_symbols, -1, -1)
        raise ValueError(f"Unsupported rolling_memory shape {tuple(rolling_memory.shape)}")


class LongTermMemoryRead(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.0, residual_init: float = 0.0, use_layer_norm: bool = True, use_gate: bool = True, gate_type: str = "scalar", read_mode: str = "gated_cross_attn", persistent_bank: PersistentMemoryBank | None = None, precomputed_encoder: nn.Module | None = None) -> None:
        super().__init__()
        if read_mode not in {"cross_attn", "gated_cross_attn"}:
            raise ValueError(f"Unsupported read_mode={read_mode}")
        if gate_type not in {"scalar", "vector"}:
            raise ValueError(f"Unsupported gate_type={gate_type}")
        self.persistent_bank = persistent_bank
        self.precomputed_encoder = precomputed_encoder
        self.use_layer_norm = use_layer_norm
        self.use_gate = use_gate
        self.gate_type = gate_type
        self.read_mode = read_mode
        self.q_ln = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        self.attn = nn.MultiheadAttention(d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        gate_out = 1 if gate_type == "scalar" else d_model
        self.gate_mlp = nn.Linear(d_model, gate_out) if use_gate else None
        self.residual_alpha = nn.Parameter(torch.tensor(float(residual_init)))
        self._last_aux: dict[str, Any] = {}
        self._last_gate: torch.Tensor | None = None

    def forward(self, h: torch.Tensor, symbol_ids: torch.Tensor | None = None, precomputed_memory: torch.Tensor | None = None) -> torch.Tensor:
        b, n, t, d = h.shape
        mem_chunks = []
        if self.persistent_bank is not None:
            mem_chunks.append(self.persistent_bank.get_memory(b, n, symbol_ids=symbol_ids, device=h.device, dtype=h.dtype))
        if precomputed_memory is not None:
            if self.precomputed_encoder is not None:
                mem_chunks.append(self.precomputed_encoder(precomputed_memory, num_symbols=n).to(device=h.device, dtype=h.dtype))
            else:
                if precomputed_memory.dim() == 3:
                    mem_chunks.append(precomputed_memory.unsqueeze(1).expand(-1, n, -1, -1).to(device=h.device, dtype=h.dtype))
                else:
                    mem_chunks.append(precomputed_memory.to(device=h.device, dtype=h.dtype))
        if not mem_chunks:
            raise ValueError("LongTermMemoryRead requires persistent bank and/or precomputed_memory")
        memory = torch.cat(mem_chunks, dim=2)
        q = self.q_ln(h)
        q2 = q.reshape(b * n, t, d)
        m2 = memory.reshape(b * n, memory.size(2), d)
        z2, attn_weights = self.attn(q2, m2, m2, need_weights=True)
        z = z2.reshape(b, n, t, d)
        if self.use_gate:
            gate = torch.sigmoid(self.gate_mlp(self.q_ln(h)))
            self._last_gate = gate
            z = z * gate
        out = h + self.residual_alpha * self.dropout(z)
        self._last_aux = {
            "long_term_memory/residual_alpha": float(self.residual_alpha.detach().item()),
            "long_term_memory/num_memory_slots": float(memory.size(2)),
        }
        if self._last_gate is not None:
            self._last_aux["long_term_memory/gate_mean"] = float(self._last_gate.detach().mean().item())
            self._last_aux["long_term_memory/gate_std"] = float(self._last_gate.detach().std().item())
        # TODO: support block placement, stateful EMA memory, Titans-style neural memory, and learned write/update.
        return out

    def get_aux_stats(self) -> dict[str, torch.Tensor | float]:
        return dict(self._last_aux)

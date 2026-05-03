from __future__ import annotations

from typing import Any

from ...utils.externals import ensure_torch

torch = ensure_torch()
nn = torch.nn
F = nn.functional


class CodebookAdapter(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_codes: int = 128,
        num_heads: int = 4,
        dropout: float = 0.0,
        topk: int | None = None,
        temperature: float = 1.0,
        residual_gate_init: float = 0.0,
        use_layernorm: bool = True,
        share_kv_codebook: bool = False,
        return_aux: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_codes = int(num_codes)
        self.num_heads = int(num_heads)
        self.topk = topk
        self.temperature = float(temperature)
        self.use_layernorm = bool(use_layernorm)
        self.share_kv_codebook = bool(share_kv_codebook)
        self.return_aux = bool(return_aux)
        self.residual_gate_init = float(residual_gate_init)

        if self.d_model <= 0 or self.num_codes <= 0 or self.num_heads <= 0:
            raise ValueError("d_model/num_codes/num_heads must be positive.")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if self.topk is not None and self.topk <= 0:
            raise ValueError("topk must be positive when provided.")

        self.head_dim = self.d_model // self.num_heads
        self.norm = nn.LayerNorm(self.d_model) if self.use_layernorm else nn.Identity()

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.attn_dropout = nn.Dropout(float(dropout))

        self.code_k = nn.Parameter(torch.empty(self.num_codes, self.d_model))
        if self.share_kv_codebook:
            self.code_v = self.code_k
        else:
            self.code_v = nn.Parameter(torch.empty(self.num_codes, self.d_model))

        self.gate = nn.Parameter(torch.tensor(self.residual_gate_init))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.code_k, mean=0.0, std=0.02)
        if not self.share_kv_codebook:
            nn.init.normal_(self.code_v, mean=0.0, std=0.02)

    def extra_repr(self) -> str:
        gate_val = float(self.gate.detach().item())
        return (
            f"d_model={self.d_model}, num_codes={self.num_codes}, num_heads={self.num_heads}, "
            f"topk={self.topk}, temperature={self.temperature}, gate_init={self.residual_gate_init}, "
            f"gate={gate_val:.6f}, share_kv_codebook={self.share_kv_codebook}"
        )

    def forward(self, x: torch.Tensor, return_aux: bool | None = None) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Expected last dim {self.d_model}, got {x.shape[-1]}.")

        want_aux = self.return_aux if return_aux is None else bool(return_aux)
        orig_shape = x.shape
        n = x.numel() // self.d_model

        y = self.norm(x).reshape(n, self.d_model)
        q = self.q_proj(y).view(n, self.num_heads, self.head_dim).transpose(0, 1)

        code_k = self.k_proj(self.code_k).view(self.num_codes, self.num_heads, self.head_dim).permute(1, 0, 2)
        code_v = self.v_proj(self.code_v).view(self.num_codes, self.num_heads, self.head_dim).permute(1, 0, 2)

        logits = torch.matmul(q, code_k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        logits = logits / max(self.temperature, 1e-6)

        if self.topk is not None and self.topk < self.num_codes:
            tk = int(self.topk)
            topk_vals, topk_idx = torch.topk(logits, k=tk, dim=-1)
            neg_inf = torch.finfo(logits.dtype).min
            masked = torch.full_like(logits, neg_inf)
            masked.scatter_(-1, topk_idx, topk_vals)
            logits = masked

        attn = F.softmax(logits, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, code_v)
        out = out.transpose(0, 1).contiguous().view(n, self.d_model)
        out = self.out_proj(out).view(orig_shape)

        x_out = x + self.gate.to(dtype=x.dtype) * out

        if not want_aux:
            return x_out

        attn_mean = attn.mean(dim=(0, 1)).to(dtype=torch.float32)
        attn_entropy = -(attn_mean * (attn_mean.clamp_min(1e-12).log())).sum()
        aux = {
            "code_attn_mean": attn_mean,
            "code_attn_entropy": attn_entropy,
            "code_effective_num": attn_entropy.exp(),
            "code_gate": self.gate.detach(),
        }
        return x_out, aux

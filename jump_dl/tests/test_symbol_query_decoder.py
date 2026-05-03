import torch

from jump_dl.src.models.layers.symbol_query_decoder import SymbolQueryDecoder


def test_shape_preservation_full_causal():
    h = torch.randn(2, 4, 6, 16)
    dec = SymbolQueryDecoder(d_model=16, num_heads=4, mode="full_causal", residual_init=1.0, use_symbol_embedding=False)
    y = dec(h)
    assert y.shape == h.shape


def _assert_no_future_leak(mode: str, topk_indices=None):
    torch.manual_seed(0)
    b, n, t, d = 2, 4, 6, 16
    h = torch.randn(b, n, t, d)
    dec = SymbolQueryDecoder(
        d_model=d,
        num_heads=4,
        mode=mode,
        residual_init=1.0,
        use_symbol_embedding=False,
        topk_indices=topk_indices,
    )
    t0 = 2
    y1 = dec(h)
    h2 = h.clone()
    h2[:, :, t0 + 1 :, :] += 100.0 * torch.randn_like(h2[:, :, t0 + 1 :, :])
    y2 = dec(h2)
    assert torch.allclose(y1[:, :, : t0 + 1, :], y2[:, :, : t0 + 1, :], atol=1e-5, rtol=1e-5)


def test_no_future_leakage_all_modes():
    topk = torch.tensor([[1, 2], [0, 2], [0, 1], [1, 2]])
    _assert_no_future_leak("full_causal")
    _assert_no_future_leak("product_memory")
    _assert_no_future_leak("topk_static", topk_indices=topk)


def test_exclude_self_behavior():
    b, n, t, d = 2, 4, 6, 16
    h = torch.randn(b, n, t, d)
    dec = SymbolQueryDecoder(d_model=d, num_heads=4, mode="full_causal", exclude_self=True, residual_init=1.0, use_symbol_embedding=False)
    y1 = dec(h)
    h2 = h.clone()
    h2[:, 0, :, :] += 250.0
    y2 = dec(h2)
    assert torch.allclose(y1[:, 0, :, :], y2[:, 0, :, :], atol=1e-5, rtol=1e-5)


def test_topk_static_isolation():
    b, n, t, d = 2, 4, 6, 16
    h = torch.randn(b, n, t, d)
    topk = torch.tensor([[1, 2], [0, 2], [0, 1], [1, 2]])
    dec = SymbolQueryDecoder(d_model=d, num_heads=4, mode="topk_static", topk_indices=topk, residual_init=1.0, use_symbol_embedding=False)
    y1 = dec(h)
    h2 = h.clone()
    h2[:, 3, :, :] += 250.0
    y2 = dec(h2)
    assert torch.allclose(y1[:, 0, :, :], y2[:, 0, :, :], atol=1e-5, rtol=1e-5)


def test_residual_initialization_zero_preserves_input():
    h = torch.randn(2, 4, 6, 16)
    dec = SymbolQueryDecoder(d_model=16, num_heads=4, mode="full_causal", residual_init=0.0, use_symbol_embedding=False)
    y = dec(h)
    assert torch.allclose(y, h)


def test_backward_pass():
    h = torch.randn(2, 4, 6, 16, requires_grad=True)
    dec = SymbolQueryDecoder(d_model=16, num_heads=4, mode="full_causal", residual_init=1.0, use_symbol_embedding=False)
    y = dec(h)
    y.sum().backward()
    assert h.grad is not None

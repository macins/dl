import torch

from jump_dl.src.models.layers.multiresolution import (
    CausalConv1dTime,
    CausalPatchMemoryCrossAttention,
    MultiScaleCausalConv,
)


def _no_leakage_check(module, x, t_future):
    y1 = module(x)
    x2 = x.clone()
    x2[..., t_future:, :] += 1000.0
    y2 = module(x2)
    torch.testing.assert_close(y1[..., :t_future, :], y2[..., :t_future, :], atol=1e-5, rtol=1e-5)


def test_causal_conv_no_leakage():
    x = torch.randn(2, 3, 20, 8)
    m = CausalConv1dTime(d_model=8, kernel_size=5)
    _no_leakage_check(m, x, 10)


def test_multiscale_no_leakage():
    x = torch.randn(2, 3, 20, 8)
    m = MultiScaleCausalConv(d_model=8, scales=[5, 15], dropout=0.0, fusion="softmax_gate")
    _no_leakage_check(m, x, 10)


def test_patch_memory_no_leakage():
    x = torch.randn(2, 3, 20, 8)
    m = CausalPatchMemoryCrossAttention(d_model=8, scales=[5], num_heads=2, dropout=0.0)
    _no_leakage_check(m, x, 10)


def test_shape_preservation():
    x3 = torch.randn(2, 20, 8)
    x4 = torch.randn(2, 3, 20, 8)
    m1 = MultiScaleCausalConv(d_model=8)
    m2 = CausalPatchMemoryCrossAttention(d_model=8, num_heads=2)
    assert m1(x3).shape == x3.shape
    assert m1(x4).shape == x4.shape
    assert m2(x3).shape == x3.shape
    assert m2(x4).shape == x4.shape


def test_backward_pass():
    x = torch.randn(2, 3, 20, 8, requires_grad=True)
    m = MultiScaleCausalConv(d_model=8)
    m(x).sum().backward()
    assert x.grad is not None

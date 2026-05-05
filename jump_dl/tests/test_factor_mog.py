import torch

from jump_dl.src.models.head.factor_mog import ConditionalLatentFactorMoGHead


def _check_common(fm, min_sigma: float):
    assert torch.isfinite(fm["pred"]).all()
    assert torch.isfinite(fm["mix_probs"]).all()
    assert torch.isfinite(fm["factor_sigma"]).all()
    assert torch.isfinite(fm["residual_sigma"]).all()
    assert torch.allclose(fm["mix_probs"].sum(dim=-1), torch.ones_like(fm["mix_probs"].sum(dim=-1)), atol=1e-5)
    assert (fm["factor_sigma"] >= min_sigma).all()
    assert (fm["residual_sigma"] >= min_sigma).all()


def test_bntd_shapes_and_backward():
    B, N, T, D, P, K = 2, 5, 7, 16, 4, 3
    h = torch.randn(B, N, T, D, requires_grad=True)
    head = ConditionalLatentFactorMoGHead(input_dim=D, num_factors=P, num_components=K, min_factor_sigma=1e-4, min_residual_sigma=1e-4)
    out = head(h)
    fm = out["factor_mog"]
    assert out["pred"].shape == (B, N, T)
    assert fm["pred"].shape == (B, N, T)
    assert fm["exposure"].shape == (B, N, T, P)
    assert fm["factor_mu"].shape == (B, T, K, P)
    assert fm["component_pred"].shape == (B, N, T, K)
    _check_common(fm, 1e-4)
    out["pred"].mean().backward()


def test_bntd_final_step_shapes():
    B, N, T, D, P, K = 2, 5, 7, 16, 4, 3
    h = torch.randn(B, N, T, D)
    head = ConditionalLatentFactorMoGHead(input_dim=D, num_factors=P, num_components=K, final_step_only=True)
    out = head(h)
    fm = out["factor_mog"]
    assert out["pred"].shape == (B, N)
    assert fm["pred"].shape == (B, N)
    assert fm["exposure"].shape == (B, N, P)
    assert fm["factor_mu"].shape == (B, K, P)
    assert fm["component_pred"].shape == (B, N, K)


def test_btd_layout_shapes():
    B, T, D, P, K = 2, 7, 16, 4, 3
    h = torch.randn(B, T, D)
    head = ConditionalLatentFactorMoGHead(input_dim=D, num_factors=P, num_components=K, three_dim_layout="BTD")
    out = head(h)
    assert out["pred"].shape == (B, T)
    assert out["factor_mog"]["pred"].shape == (B, 1, T)


def test_btd_layout_final_step_shapes():
    B, T, D, P, K = 2, 7, 16, 4, 3
    h = torch.randn(B, T, D)
    head = ConditionalLatentFactorMoGHead(input_dim=D, num_factors=P, num_components=K, three_dim_layout="BTD", final_step_only=True)
    out = head(h)
    assert out["pred"].shape == (B,)
    assert out["factor_mog"]["pred"].shape == (B, 1)


def test_bnd_layout_shapes():
    B, N, D, P, K = 2, 5, 16, 4, 3
    h = torch.randn(B, N, D)
    head = ConditionalLatentFactorMoGHead(input_dim=D, num_factors=P, num_components=K, three_dim_layout="BND")
    out = head(h)
    assert out["pred"].shape == (B, N)
    assert out["factor_mog"]["pred"].shape == (B, N, 1)


def test_bd_shapes():
    B, D, P, K = 2, 16, 4, 3
    h = torch.randn(B, D)
    out = ConditionalLatentFactorMoGHead(input_dim=D, num_factors=P, num_components=K, final_step_only=False)(h)
    assert out["pred"].shape == (B,)
    assert out["factor_mog"]["pred"].shape == (B, 1, 1)

    out2 = ConditionalLatentFactorMoGHead(input_dim=D, num_factors=P, num_components=K, final_step_only=True)(h)
    assert out2["pred"].shape == (B,)
    assert out2["factor_mog"]["pred"].shape == (B, 1)

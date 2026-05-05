import torch

from jump_dl.src.models import build_model


def _build(**kwargs):
    cfg = {
        "name": "encoder_decoder_sequence_regressor",
        "input_dim": 32,
        "d_model": 64,
        "d_ff": 128,
        "n_heads": 4,
        "dropout": 0.0,
        "n_local_layers": 2,
        "n_decoder_layers": 1,
        "num_horizons": 1,
        "use_product_memory": True,
        "use_market_memory": True,
        "use_factor_memory": True,
        "n_product_layers": 1,
        "n_market_layers": 1,
        "n_factor_temporal_layers": 1,
        "num_factor_tokens": 4,
        "factor_cross_n_heads": 4,
        "target_key": "ret_30min",
    }
    cfg.update(kwargs)
    return build_model(cfg)


def test_shapes_and_backward_single_horizon():
    m = _build(output_mode="single_horizon")
    x = torch.randn(2, 5, 16, 32)
    batch = {"features": {"continuous": x}, "product_ids": torch.tensor([0, 0, 1, 1, 2])}
    out = m(batch)
    y = out["preds"]["ret_30min"]
    assert y.shape == (2, 5, 16)
    y.sum().backward()


def test_shapes_multi_horizon_and_disable_memories():
    m = _build(num_horizons=3, output_mode="multi_horizon", use_product_memory=False, use_factor_memory=False)
    x = torch.randn(2, 5, 16, 32)
    out = m({"features": {"continuous": x}})
    y = out["preds"]["ret_30min"]
    assert y.shape == (2, 5, 16, 3)


def test_causality_no_future_leakage():
    torch.manual_seed(0)
    m = _build()
    m.eval()
    x = torch.randn(2, 5, 16, 32)
    y1 = m({"features": {"continuous": x}})["preds"]["ret_30min"]
    x2 = x.clone()
    t0 = 8
    x2[:, :, t0 + 1 :, :] += 50.0 * torch.randn_like(x2[:, :, t0 + 1 :, :])
    y2 = m({"features": {"continuous": x2}})["preds"]["ret_30min"]
    assert torch.allclose(y1[:, :, : t0 + 1], y2[:, :, : t0 + 1], atol=1e-4, rtol=1e-4)

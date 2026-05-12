import torch

from jump_dl.src.models import build_model


def _build_model(innovation_enabled: bool):
    return build_model(
        {
            "name": "transformer_panel_regressor",
            "numeric_feature_groups": ["continuous"],
            "categorical_cols": [],
            "num_features": 8,
            "hidden_size": 16,
            "num_layers": 1,
            "dropout": 0.0,
            "target_key": "ret_30min",
            "backbone": {
                "name": "transformer_sequence",
                "hidden_size": 16,
                "num_layers": 1,
                "dropout": 0.0,
                "num_heads": 4,
                "causal": True,
                "position_encoding": "alibi",
            },
            "innovation": {
                "enabled": innovation_enabled,
                "prior_type": "gru",
                "fusion_type": "mlp",
                "aux_loss_weight": 0.01,
            },
        }
    )


def test_transformer_panel_forward_with_and_without_innovation():
    x = torch.randn(2, 3, 6, 8)
    batch = {"features": {"continuous": x}, "padding_mask": torch.ones(2, 3, 6, dtype=torch.bool)}

    model_off = _build_model(False)
    out_off = model_off(batch)
    pred_off = out_off["preds"]["ret_30min"]
    assert pred_off.shape == (2, 3, 6)

    model_on = _build_model(True)
    out_on = model_on(batch)
    pred_on = out_on["preds"]["ret_30min"]
    assert pred_on.shape == pred_off.shape

    aux = out_on["aux"]["innovation"]
    for key in ["z_pred", "log_s", "innovation", "innovation_std"]:
        assert torch.isfinite(aux[key]).all()
    assert torch.isfinite(out_on["aux_losses"]["innovation_aux_nll"])


def test_innovation_prior_no_current_token_leakage():
    torch.manual_seed(1)
    model = _build_model(True)
    model.eval()
    x = torch.randn(1, 2, 5, 8)
    batch = {"features": {"continuous": x}, "padding_mask": torch.ones(1, 2, 5, dtype=torch.bool)}
    z_pred_1 = model(batch)["aux"]["innovation"]["z_pred"].detach()

    x2 = x.clone()
    x2[:, :, 3, :] += 100.0
    z_pred_2 = model({"features": {"continuous": x2}, "padding_mask": batch["padding_mask"]})["aux"]["innovation"]["z_pred"].detach()

    # z_pred at time t can only depend on z_{<t}; changing z_t should not change z_pred_t.
    assert torch.allclose(z_pred_1[:, :, 3, :], z_pred_2[:, :, 3, :], atol=1e-5, rtol=1e-5)

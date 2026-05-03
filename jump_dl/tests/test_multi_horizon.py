import torch

from jump_dl.src.models.head.multi_horizon import MultiHorizonHeads, HorizonQueryDecoder
from jump_dl.src.objectives import CosineSimilarityObjective


def test_multi_horizon_heads_shape():
    h = torch.randn(2, 3, 7, 16)
    hz = [5, 10, 15, 30, 60]
    m = MultiHorizonHeads(16, hz)
    out = m(h)
    assert set(out.keys()) == set(hz)
    for v in out.values():
        assert v.shape == (2, 3, 7, 1)


def test_multi_horizon_objective_and_weights():
    pred = {h: torch.randn(2, 3, 7) for h in [5, 10, 30]}
    batch = {
        "targets": {"ret_5min": torch.randn(2, 3, 7), "ret_10min": torch.randn(2, 3, 7), "ret_30min": torch.randn(2, 3, 7)},
        "padding_mask": torch.ones(2, 3, 7, dtype=torch.bool),
    }
    obj = CosineSimilarityObjective(multi_horizon={"enabled": True, "horizons": [5, 10, 30], "main_horizon": 30, "label_keys": {5: "ret_5min", 10: "ret_10min", 30: "ret_30min"}, "loss": {"type": "global_cosine", "main_weight": 1.0, "aux_weights": {5: 0.1, 10: 0.2}}})
    out = obj({"pred_by_horizon": pred, "preds": {"ret_30min": pred[30]}}, batch)
    assert out.loss.ndim == 0


def test_aux_schedule():
    obj = CosineSimilarityObjective(multi_horizon={"enabled": True, "horizons": [30], "main_horizon": 30, "label_keys": {30: "ret_30min"}, "loss": {"aux_schedule": {"enabled": True, "start_weight_multiplier": 2.0, "final_weight_multiplier": 0.2, "start_step": 0, "end_step": 100}}})
    obj.global_step = 0
    assert abs(obj._aux_multiplier() - 2.0) < 1e-6
    obj.global_step = 100
    assert abs(obj._aux_multiplier() - 0.2) < 1e-6


def test_horizon_query_decoder_causality():
    torch.manual_seed(0)
    h = torch.randn(2, 2, 8, 16)
    dec = HorizonQueryDecoder(16, [5, 30], num_heads=4)
    y1, _ = dec(h)
    h2 = h.clone()
    t0 = 4
    h2[:, :, t0 + 1 :, :] += 100.0 * torch.randn_like(h2[:, :, t0 + 1 :, :])
    y2, _ = dec(h2)
    for hz in [5, 30]:
        assert torch.allclose(y1[hz][:, :, : t0 + 1], y2[hz][:, :, : t0 + 1], atol=1e-4, rtol=1e-4)

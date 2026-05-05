import torch

from jump_dl.src.models.head.factor_mog import ConditionalLatentFactorMoGHead
from jump_dl.src.objectives import FactorMoGWithAuxObjective, marginal_factor_mog_nll, exposure_orthogonality_loss


def test_factor_mog_shapes_and_backward():
    b,n,t,d,p,k = 2,5,7,16,4,3
    h = torch.randn(b,n,t,d, requires_grad=True)
    y = torch.randn(b,n,t)
    head = ConditionalLatentFactorMoGHead(input_dim=d, num_factors=p, num_components=k)
    out = head(h)
    fm = out['factor_mog']
    assert out['pred'].shape == (b,n,t)
    assert fm['component_pred'].shape == (b,n,t,k)
    assert fm['factor_mu'].shape == (b,t,k,p)
    assert torch.isfinite(out['pred']).all()
    obj = FactorMoGWithAuxObjective(lambda_mog_nll=0.01, lambda_exposure_orth=0.001, lambda_mix_entropy=0.001)
    batch={'targets': y, 'padding_mask': torch.ones_like(y, dtype=torch.bool)}
    loss = obj(out, batch).loss
    assert torch.isfinite(loss)
    loss.backward()


def test_factor_mog_final_step_and_single_component():
    b,n,t,d,p = 2,5,7,16,4
    h = torch.randn(b,n,t,d)
    y = torch.randn(b,n)
    head = ConditionalLatentFactorMoGHead(input_dim=d, num_factors=p, num_components=1, final_step_only=True)
    out = head(h)
    assert out['pred'].shape == (b,n)
    fm = out['factor_mog']
    assert fm['component_pred'].shape == (b,n,1)
    nll,_ = marginal_factor_mog_nll(y, fm['exposure'], fm['factor_mu'], fm['factor_sigma'], fm['residual_sigma'], fm['mix_logits'])
    assert torch.isfinite(nll)
    orth = exposure_orthogonality_loss(fm['exposure'], mask=torch.ones_like(y, dtype=torch.bool))
    assert torch.isfinite(orth)


def test_lambda_zero_does_not_require_factor_fields():
    obj = FactorMoGWithAuxObjective(lambda_mog_nll=0.0, lambda_exposure_orth=0.0, lambda_mix_entropy=0.0)
    pred = torch.randn(2,5,7)
    out = {'preds': {'ret_30min': pred}, 'pred': pred}
    batch={'targets': torch.randn(2,5,7), 'padding_mask': torch.ones(2,5,7, dtype=torch.bool)}
    result = obj(out, batch)
    assert torch.isfinite(result.loss)

import torch

from jump_dl.src.models.layers.codebook import CodebookAdapter
from jump_dl.src.models.layers.transformer import TransformerEncoderBlock


def test_codebook_adapter_shapes_and_aux():
    x = torch.randn(2, 16, 32)
    adapter = CodebookAdapter(d_model=32, num_codes=16, num_heads=4, topk=None, return_aux=True)
    y, aux = adapter(x)
    assert y.shape == x.shape
    for key in ["code_attn_mean", "code_attn_entropy", "code_effective_num", "code_gate"]:
        assert key in aux
    assert aux["code_attn_mean"].shape == (16,)


def test_codebook_adapter_topk_and_backward():
    x = torch.randn(2, 16, 32, requires_grad=True)
    adapter = CodebookAdapter(d_model=32, num_codes=16, num_heads=4, topk=8)
    y = adapter(x).sum()
    y.backward()
    assert adapter.code_k.grad is not None
    if adapter.share_kv_codebook:
        assert adapter.code_k.grad is not None
    else:
        assert adapter.code_v.grad is not None


def test_transformer_block_compat_when_codebook_disabled():
    x = torch.randn(2, 16, 32)
    block = TransformerEncoderBlock(hidden_size=32, num_heads=4, codebook_enabled=False)
    y = block(x)
    assert y.shape == x.shape

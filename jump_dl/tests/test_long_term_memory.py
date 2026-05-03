import torch

from jump_dl.src.models.layers.long_term_memory import LongTermMemoryRead, PersistentMemoryBank, PrecomputedMemoryEncoder


def test_persistent_memory_shape_market_only():
    b, n, t, d = 2, 3, 5, 16
    h = torch.randn(b, n, t, d)
    bank = PersistentMemoryBank(d_model=d, num_market_slots=4, memory_levels=["market"])
    ltm = LongTermMemoryRead(d_model=d, num_heads=4, residual_init=0.1, persistent_bank=bank)
    out = ltm(h)
    assert out.shape == h.shape


def test_market_symbol_memory_shape():
    b, n, t, d = 2, 4, 5, 16
    h = torch.randn(b, n, t, d)
    ids = torch.arange(n)
    bank = PersistentMemoryBank(d_model=d, num_market_slots=4, num_symbol_slots=2, num_symbols=8, memory_levels=["market", "symbol"])
    ltm = LongTermMemoryRead(d_model=d, num_heads=4, residual_init=0.1, persistent_bank=bank)
    out = ltm(h, symbol_ids=ids)
    assert out.shape == h.shape


def test_residual_zero_preserves_input():
    b, n, t, d = 2, 3, 5, 16
    h = torch.randn(b, n, t, d)
    bank = PersistentMemoryBank(d_model=d, num_market_slots=4, memory_levels=["market"])
    ltm = LongTermMemoryRead(d_model=d, num_heads=4, residual_init=0.0, persistent_bank=bank)
    out = ltm(h)
    assert torch.allclose(out, h, atol=1e-7, rtol=1e-7)


def test_backward_pass_grads_exist():
    b, n, t, d = 2, 3, 5, 16
    h = torch.randn(b, n, t, d, requires_grad=True)
    bank = PersistentMemoryBank(d_model=d, num_market_slots=4, memory_levels=["market"])
    ltm = LongTermMemoryRead(d_model=d, num_heads=4, residual_init=0.1, persistent_bank=bank)
    out = ltm(h)
    out.sum().backward()
    assert bank.market_memory.grad is not None
    assert ltm.attn.in_proj_weight.grad is not None


def test_precomputed_encoder_shapes():
    b, n, l, s, d = 2, 3, 4, 6, 8
    enc = PrecomputedMemoryEncoder(summary_dim=s, d_model=d, num_summary_slots=l, pooling="none")
    x1 = torch.randn(b, n, l, s)
    y1 = enc(x1, n)
    assert y1.shape == (b, n, l, d)
    x2 = torch.randn(b, l, s)
    y2 = enc(x2, n)
    assert y2.shape == (b, n, l, d)
    x3 = torch.randn(b, n, 5, d)
    y3 = enc(x3, n)
    assert y3.shape == x3.shape


def test_gate_stats_scalar_vector():
    b, n, t, d = 2, 3, 5, 16
    h = torch.randn(b, n, t, d)
    bank = PersistentMemoryBank(d_model=d, num_market_slots=4, memory_levels=["market"])
    ltm_s = LongTermMemoryRead(d_model=d, num_heads=4, residual_init=0.1, persistent_bank=bank, gate_type="scalar")
    _ = ltm_s(h)
    assert ltm_s._last_gate is not None and ltm_s._last_gate.shape[-1] == 1
    stats = ltm_s.get_aux_stats()
    assert "long_term_memory/residual_alpha" in stats and "long_term_memory/gate_mean" in stats

    ltm_v = LongTermMemoryRead(d_model=d, num_heads=4, residual_init=0.1, persistent_bank=bank, gate_type="vector")
    _ = ltm_v(h)
    assert ltm_v._last_gate is not None and ltm_v._last_gate.shape[-1] == d


def test_symbol_memory_gather_permutation_changes_order():
    d = 4
    bank = PersistentMemoryBank(d_model=d, num_market_slots=0, num_symbol_slots=1, num_symbols=4, memory_levels=["symbol"])
    with torch.no_grad():
        for i in range(4):
            bank.symbol_memory[i, 0, :] = float(i)
    ids_a = torch.tensor([0, 1, 2, 3])
    ids_b = torch.tensor([3, 2, 1, 0])
    mem_a = bank.get_memory(batch_size=1, num_symbols=4, symbol_ids=ids_a)
    mem_b = bank.get_memory(batch_size=1, num_symbols=4, symbol_ids=ids_b)
    assert not torch.allclose(mem_a, mem_b)
    assert torch.allclose(mem_b[0, 0, 0], torch.full((d,), 3.0))

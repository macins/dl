from __future__ import annotations

import torch

from jump_dl.src.optim.nexus import NexusEngine


class TinyMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 4),
        )

    def forward(self, batch):
        return self.net(batch["x"])


def _run_once(scope: str) -> None:
    torch.manual_seed(0)
    model = TinyMLP()
    outer_opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    nexus = NexusEngine(model=model, inner_lr=1e-2, normalize_scope=scope)
    nexus.sync_inner_from_main()

    for p0, p1 in zip(model.parameters(), nexus.inner_model.parameters()):
        assert torch.allclose(p0, p1)

    batch = {"x": torch.randn(4, 8), "y": torch.randn(4, 4)}
    inner_before = [p.detach().clone() for p in nexus.inner_model.parameters()]
    main_before = [p.detach().clone() for p in model.parameters()]

    out = nexus.inner_model(batch)
    loss = torch.nn.functional.mse_loss(out, batch["y"])
    loss.backward()
    nexus.inner_step()
    nexus.zero_inner_grad()

    assert any(not torch.allclose(a, b) for a, b in zip(inner_before, nexus.inner_model.parameters()))
    assert all(torch.allclose(a, b) for a, b in zip(main_before, model.parameters()))

    nexus.assign_pseudo_grad_to_main()
    for pm, pi in zip(model.parameters(), nexus.inner_model.parameters()):
        assert pm.grad is not None
        assert torch.allclose(pm.grad, pm.detach() - pi.detach(), atol=1e-6)

    outer_opt.step()
    assert any(not torch.allclose(a, b) for a, b in zip(main_before, model.parameters()))

    nexus.sync_inner_from_main()
    for p0, p1 in zip(model.parameters(), nexus.inner_model.parameters()):
        assert torch.allclose(p0, p1)


def test_nexus_global() -> None:
    _run_once("global")


def test_nexus_per_param() -> None:
    _run_once("per_param")

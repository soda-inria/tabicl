"""AMP setup must work across the supported PyTorch versions."""

import pytest
import torch

from tabicl import FinetunedTabICLClassifier, FinetunedTabICLRegressor


@pytest.mark.parametrize("estimator_cls", [FinetunedTabICLClassifier, FinetunedTabICLRegressor])
@pytest.mark.parametrize("amp", [False, True])
def test_cpu_optimizer_step_without_top_level_grad_scaler(estimator_cls, amp, monkeypatch):
    # PyTorch 2.2 exposes GradScaler only through torch.cuda.amp.
    monkeypatch.delattr(torch, "GradScaler", raising=False)
    estimator = estimator_cls(amp=amp)
    use_amp, scaler, context = estimator._make_amp(torch.device("cpu"))
    assert not use_amp
    assert not scaler.is_enabled()

    parameter = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    with context():
        loss = parameter.square()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    scaler.step(optimizer)
    scaler.update()
    torch.testing.assert_close(parameter, torch.tensor(1.6))


@pytest.mark.parametrize("amp", [False, True])
def test_legacy_cuda_scaler_retains_amp_setting(amp, monkeypatch):
    monkeypatch.delattr(torch, "GradScaler", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    estimator = FinetunedTabICLClassifier(amp=amp)
    use_amp, scaler, _ = estimator._make_amp(torch.device("cuda"))
    # Constructing a scaler is lazy and does not allocate GPU tensors.
    assert use_amp is amp
    assert scaler.is_enabled() is amp


def test_modern_grad_scaler_is_preferred(monkeypatch):
    calls = []
    legacy_scaler = torch.cuda.amp.GradScaler

    def modern_scaler(device, *, enabled):
        calls.append((device, enabled))
        return legacy_scaler(enabled=enabled)

    monkeypatch.setattr(torch, "GradScaler", modern_scaler, raising=False)
    _, scaler, _ = FinetunedTabICLClassifier()._make_amp(torch.device("cpu"))
    assert calls == [("cuda", False)]
    assert not scaler.is_enabled()

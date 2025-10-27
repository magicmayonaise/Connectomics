import pytest

torch = pytest.importorskip("torch")

from connectomics.dyn_opt import ActivationMaximizer, toy_slice


def _to_tensor(matrix):
    return torch.tensor(matrix, dtype=torch.float32)


def test_gradients_exist():
    weights = _to_tensor(toy_slice())
    maximizer = ActivationMaximizer(weights, depth=3)
    params = torch.zeros(
        weights.shape[0], dtype=weights.dtype, device=weights.device, requires_grad=True
    )
    output = maximizer.forward(params)
    target_value = output[-1]
    target_value.backward()
    assert params.grad is not None
    assert torch.all(torch.isfinite(params.grad))


def test_activation_maximization_improves_target():
    weights = _to_tensor(toy_slice(seed=1))
    maximizer = ActivationMaximizer(weights, depth=4)
    baseline = maximizer.forward(
        torch.zeros(weights.shape[0], dtype=weights.dtype, device=weights.device)
    )[-1].item()
    history = maximizer.maximize(target_index=weights.shape[0] - 1, steps=25, lr=0.2)
    assert history[-1] > baseline
    assert history[-1] >= history[0]

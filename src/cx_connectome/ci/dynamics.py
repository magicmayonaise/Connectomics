"""Minimal dynamics model for circuit inference simulations."""
from __future__ import annotations

def _require_torch():
    """Import :mod:`torch` lazily and raise a friendly error if unavailable."""
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in environments without torch
        raise RuntimeError(
            "The 'torch' package is required to run circuit dynamics simulations. "
            "Install it with 'pip install torch' to enable this functionality."
        ) from exc

    return torch


def simulate(
    W,
    inputs,
    T: int,
    tau: float,
    excitability: float,
    divisive_norm: bool = False,
    bias=None,
):
    """Simulate the non-linear dynamics of a recurrent network.

    Parameters
    ----------
    W:
        Recurrent weight matrix (``N x N``) or array-like structure convertible to
        a :class:`torch.Tensor`.
    inputs:
        Time-varying external input sequence of length ``T`` where each entry has
        dimensionality ``N``. Scalar or vector inputs are broadcast as needed. If
        ``None`` a zero input sequence is used.
    T:
        Number of simulation steps.
    tau:
        Leak parameter controlling the contribution of the previous state.
    excitability:
        Scaling applied to the synaptic drive before the non-linearities.
    divisive_norm:
        If ``True`` apply divisive normalization to the synaptic drive prior to
        the non-linearities.
    bias:
        Optional constant bias added at every time step.

    Returns
    -------
    torch.Tensor
        Simulated activity trajectory of shape ``(T + 1, N)``. The first entry is
        the initial activity state (zeros).
    """

    if T <= 0:
        raise ValueError("T must be a positive integer")

    torch = _require_torch()

    W_tensor = torch.as_tensor(W, dtype=torch.float32)
    if W_tensor.ndim != 2 or W_tensor.shape[0] != W_tensor.shape[1]:
        raise ValueError("W must be a square matrix")

    n_units = W_tensor.shape[0]

    device = W_tensor.device

    if inputs is None:
        input_tensor = torch.zeros((T, n_units), dtype=W_tensor.dtype, device=device)
    else:
        input_tensor = torch.as_tensor(inputs, dtype=W_tensor.dtype, device=device)
        if input_tensor.ndim == 0:
            input_tensor = input_tensor.repeat(T, n_units)
        elif input_tensor.ndim == 1:
            if input_tensor.shape[0] == n_units:
                input_tensor = input_tensor.view(1, n_units).repeat(T, 1)
            elif input_tensor.shape[0] == T:
                input_tensor = input_tensor.view(T, 1).repeat(1, n_units)
            elif input_tensor.shape[0] == 1:
                input_tensor = input_tensor.view(1, 1).repeat(T, n_units)
            else:
                raise ValueError(
                    "1D inputs must have length T, number of units, or be scalar"
                )
        else:
            if input_tensor.shape[0] != T and input_tensor.shape[-1] == T:
                input_tensor = input_tensor.transpose(0, -1)
            if input_tensor.shape[0] != T:
                raise ValueError("Input sequence must provide T time steps")
            input_tensor = input_tensor.reshape(T, -1)
            if input_tensor.shape[1] == 1 and n_units != 1:
                input_tensor = input_tensor.repeat(1, n_units)
            if input_tensor.shape[1] != n_units:
                raise ValueError("Each input vector must match the network dimension")

    if bias is None:
        bias_tensor = torch.zeros(n_units, dtype=W_tensor.dtype, device=device)
    else:
        bias_tensor = torch.as_tensor(bias, dtype=W_tensor.dtype, device=device)
        if bias_tensor.ndim == 0:
            bias_tensor = bias_tensor.repeat(n_units)
        else:
            bias_tensor = bias_tensor.view(-1)
            if bias_tensor.numel() == 1 and n_units != 1:
                bias_tensor = bias_tensor.repeat(n_units)
            if bias_tensor.numel() != n_units:
                raise ValueError("Bias must be broadcastable to the number of units")

    states = torch.zeros((T + 1, n_units), dtype=W_tensor.dtype, device=device)

    for t in range(T):
        a_t = states[t]
        drive = torch.matmul(W_tensor, a_t) + input_tensor[t]

        if divisive_norm:
            norm = drive.norm(p=2).clamp_min(1e-6)
            drive = drive / (1.0 + norm)

        pre_activation = excitability * drive + tau * a_t + bias_tensor
        activity = torch.tanh(torch.relu(pre_activation))
        states[t + 1] = activity

    return states


__all__ = ["simulate"]

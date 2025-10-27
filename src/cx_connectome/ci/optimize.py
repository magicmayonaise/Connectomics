"""Optimization utilities for connectome interpreters.

This module implements a lightweight activation maximisation routine used to
propose inputs that drive a set of *target* neurons.  The optimisation is kept
purely in Python so that no external numerical dependencies are required.  The
main entry point :func:`optimal_input` accepts a connectivity matrix and lists
of node identifiers and returns both the optimised input program and a rich
record of the optimisation trace.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import sqrt
from numbers import Integral
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

MatrixLike = Union[Sequence[Sequence[float]], "SupportsToList"]


class SupportsToList:
    """Protocol-like helper used for duck-typing inputs with ``tolist``."""

    def tolist(self) -> Sequence[Sequence[float]]:  # pragma: no cover - documentation only
        raise NotImplementedError


@dataclass
class OptimisationRecord:
    """Stores intermediate values from a single optimisation step."""

    step: int
    loss: float
    objective: float
    l2_penalty: float
    sparsity_penalty: float


def _coerce_matrix(data: MatrixLike) -> List[List[float]]:
    if hasattr(data, "tolist"):
        raw = data.tolist()  # type: ignore[assignment]
    else:
        raw = data  # type: ignore[assignment]
    if not isinstance(raw, Sequence):
        raise TypeError("Matrix must be a sequence of sequences")
    rows: List[List[float]] = []
    for row in raw:
        if not isinstance(row, Sequence):
            raise TypeError("Matrix rows must be sequences")
        rows.append([float(value) for value in row])
    if not rows:
        raise ValueError("Matrix is empty")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("Matrix rows must all be the same length")
    return rows


def _ensure_matrix(
    matrix: Union[
        MatrixLike,
        Tuple[MatrixLike, Sequence[Union[int, str]]],
        Mapping[str, MatrixLike],
    ]
) -> Tuple[List[List[float]], List[str]]:
    if isinstance(matrix, tuple) and len(matrix) == 2:
        array, labels = matrix
        rows = _coerce_matrix(array)
        labels_list = [str(item) for item in labels]
        if len(labels_list) != len(rows):
            raise ValueError("Number of labels must match matrix dimensions")
        if len(rows) != len(rows[0]):
            raise ValueError("Connectivity matrix must be square")
        return rows, labels_list

    if hasattr(matrix, "values") and hasattr(matrix, "index"):
        values = getattr(matrix, "values")
        index = getattr(matrix, "index")
        rows = _coerce_matrix(values)
        labels = [str(item) for item in index]
        if len(labels) != len(rows):
            raise ValueError("Number of index labels must match matrix dimensions")
        if len(rows) != len(rows[0]):
            raise ValueError("Connectivity matrix must be square")
        return rows, labels

    rows = _coerce_matrix(matrix)
    if len(rows) != len(rows[0]):
        raise ValueError("Connectivity matrix must be square")
    labels = [str(i) for i in range(len(rows))]
    return rows, labels


def _ids_to_indices(ids: Iterable[Union[int, str]], mapping: Mapping[str, int]) -> List[int]:
    indices: List[int] = []
    for item in ids:
        if isinstance(item, Integral):
            indices.append(int(item))
            continue
        key = str(item)
        if key not in mapping:
            raise KeyError(f"Unknown node identifier: {item!r}")
        indices.append(mapping[key])
    return indices


@dataclass
class OptimalInputResult:
    """Result container for :func:`optimal_input`.

    Attributes
    ----------
    input_series:
        Optimised time-series with shape ``(len(input_ids), T)`` represented as
        a list-of-lists.  The outer list enumerates input channels, the inner
        list enumerates time steps.
    activations:
        State trajectory with shape ``(T + 1, n_nodes)`` as a list of vectors.
    target_trace:
        Activation of target neurons along the trajectory with shape
        ``(T + 1, len(target_ids))``.
    loss_history:
        Sequence of :class:`OptimisationRecord` describing optimisation
        progress.
    labels:
        Ordered node labels used internally during optimisation.
    metadata:
        Miscellaneous information that can aid downstream visualisation.
    """

    input_series: List[List[float]]
    activations: List[List[float]]
    target_trace: List[List[float]]
    loss_history: List[OptimisationRecord]
    labels: List[str]
    metadata: Dict[str, object]


def _vector_zero(size: int) -> List[float]:
    return [0.0] * size


def _matrix_vector_product(matrix: List[List[float]], vector: List[float]) -> List[float]:
    return [sum(row[j] * vector[j] for j in range(len(vector))) for row in matrix]


def _transpose(matrix: List[List[float]]) -> List[List[float]]:
    return [[matrix[row][col] for row in range(len(matrix))] for col in range(len(matrix[0]))]


def _vector_norm(vector: List[float]) -> float:
    return sqrt(sum(value * value for value in vector))


def _mean(values: Iterable[float]) -> float:
    total = 0.0
    count = 0
    for value in values:
        total += value
        count += 1
    return total / count if count else 0.0


def _sign(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def optimal_input(
    W: Union[
        MatrixLike,
        Tuple[MatrixLike, Sequence[Union[int, str]]],
        Mapping[str, MatrixLike],
    ],
    target_ids: Sequence[Union[int, str]],
    input_ids: Sequence[Union[int, str]],
    T: int,
    loss: str = "mean_target",
    l2: float = 1e-3,
    sparsity: Optional[float] = None,
    *,
    steps: int = 1000,
    lr: float = 0.05,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    signed: bool = False,
    divisive_norm: bool = False,
    initial_program: Optional[Sequence[Sequence[float]]] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
    callback: Optional[Callable[[OptimalInputResult], None]] = None,
) -> OptimalInputResult:
    if T <= 0:
        raise ValueError("T must be a positive integer")

    matrix, labels = _ensure_matrix(W)
    node_lookup = {label: idx for idx, label in enumerate(labels)}
    target_indices = _ids_to_indices(target_ids, node_lookup)
    input_indices = _ids_to_indices(input_ids, node_lookup)

    if not target_indices:
        raise ValueError("At least one target id is required")
    if not input_indices:
        raise ValueError("At least one input id is required")

    n_nodes = len(matrix)
    n_inputs = len(input_indices)

    if initial_program is not None:
        program = [[float(value) for value in row] for row in initial_program]
        if len(program) != n_inputs or any(len(row) != T for row in program):
            raise ValueError("initial_program must have shape (len(input_ids), T)")
    else:
        rng = random.Random(seed)
        program = [[0.0 for _ in range(T)] for _ in range(n_inputs)]
        if seed is not None:
            for i in range(n_inputs):
                for t in range(T):
                    program[i][t] = 1e-3 * rng.gauss(0.0, 1.0)

    m1 = [[0.0 for _ in range(T)] for _ in range(n_inputs)]
    m2 = [[0.0 for _ in range(T)] for _ in range(n_inputs)]

    regularisation_l2 = float(l2)
    regularisation_l1 = float(sparsity) if sparsity is not None else None

    records: List[OptimisationRecord] = []
    matrix_T = _transpose(matrix)

    def forward(u: List[List[float]]) -> Tuple[List[List[float]], List[List[float]], List[List[bool]], List[float]]:
        states = [[0.0 for _ in range(n_nodes)] for _ in range(T + 1)]
        pre_activation = [[0.0 for _ in range(n_nodes)] for _ in range(T)]
        relu_mask = [[False for _ in range(n_nodes)] for _ in range(T)]
        norms = [1.0 for _ in range(T)]

        for t in range(T):
            raw = _matrix_vector_product(matrix, states[t])
            for col, idx in enumerate(input_indices):
                raw[idx] += u[col][t]
            pre_activation[t] = raw[:]
            if divisive_norm:
                norm = _vector_norm(raw) + 1e-6
                norms[t] = norm
                raw = [value / norm for value in raw]
            if not signed:
                mask = [value > 0.0 for value in raw]
                relu_mask[t] = mask
                raw = [value if mask_i else 0.0 for value, mask_i in zip(raw, mask)]
            states[t + 1] = raw[:]
        return states, pre_activation, relu_mask, norms

    def backward(
        grad_final: List[float],
        pre_activation: List[List[float]],
        relu_mask: List[List[bool]],
        norms: List[float],
    ) -> List[List[float]]:
        grad_program = [[0.0 for _ in range(T)] for _ in range(n_inputs)]
        grad_state = grad_final[:]

        for t in range(T - 1, -1, -1):
            grad = grad_state[:]
            if not signed:
                grad = [value if mask else 0.0 for value, mask in zip(grad, relu_mask[t])]
            raw = pre_activation[t]
            if divisive_norm:
                norm = norms[t]
                inv_norm = 1.0 / norm
                dot = sum(r * g for r, g in zip(raw, grad))
                grad = [g * inv_norm - r * (dot * (inv_norm ** 3)) for r, g in zip(raw, grad)]
            for col, idx in enumerate(input_indices):
                grad_program[col][t] = grad[idx]
            grad_state = _matrix_vector_product(matrix_T, grad)
        return grad_program

    beta1_pow = 1.0
    beta2_pow = 1.0

    for step in range(1, steps + 1):
        states, pre_activation, relu_mask, norms = forward(program)
        target_trace = [[state[idx] for idx in target_indices] for state in states]
        if loss != "mean_target":
            raise NotImplementedError("Only 'mean_target' loss is implemented")

        objective = _mean(target_trace[-1])
        grad_final = _vector_zero(n_nodes)
        for idx in target_indices:
            grad_final[idx] = -1.0 / len(target_indices)

        grad_program = backward(grad_final, pre_activation, relu_mask, norms)

        l2_penalty = 0.0
        l1_penalty = 0.0
        if regularisation_l2:
            sq_sum = sum(value * value for row in program for value in row)
            l2_penalty = 0.5 * regularisation_l2 * (sq_sum / (n_inputs * T))
            for i in range(n_inputs):
                for t in range(T):
                    grad_program[i][t] += regularisation_l2 * program[i][t]
        if regularisation_l1 is not None:
            abs_sum = sum(abs(value) for row in program for value in row)
            l1_penalty = regularisation_l1 * (abs_sum / (n_inputs * T))
            for i in range(n_inputs):
                for t in range(T):
                    grad_program[i][t] += regularisation_l1 * _sign(program[i][t])

        loss_value = -objective + l2_penalty + l1_penalty

        for i in range(n_inputs):
            for t in range(T):
                grad = grad_program[i][t]
                m1[i][t] = beta1 * m1[i][t] + (1.0 - beta1) * grad
                m2[i][t] = beta2 * m2[i][t] + (1.0 - beta2) * (grad * grad)

        beta1_pow *= beta1
        beta2_pow *= beta2
        m1_hat_denom = 1.0 - beta1_pow
        m2_hat_denom = 1.0 - beta2_pow

        for i in range(n_inputs):
            for t in range(T):
                m1_hat = m1[i][t] / m1_hat_denom
                m2_hat = m2[i][t] / m2_hat_denom
                update = lr * m1_hat / (sqrt(m2_hat) + eps)
                program[i][t] -= update
                if not signed and program[i][t] < 0.0:
                    program[i][t] = 0.0

        record = OptimisationRecord(
            step=step,
            loss=float(loss_value),
            objective=float(objective),
            l2_penalty=float(l2_penalty),
            sparsity_penalty=float(l1_penalty),
        )
        records.append(record)

        if verbose and (step == 1 or step % 100 == 0):
            print(
                f"[optimal_input] step={step:04d} loss={record.loss:.4f} "
                f"objective={record.objective:.4f}"
            )

        if callback is not None:
            callback(
                OptimalInputResult(
                    input_series=[row[:] for row in program],
                    activations=[state[:] for state in states],
                    target_trace=[trace[:] for trace in target_trace],
                    loss_history=records.copy(),
                    labels=labels[:],
                    metadata={
                        "target_ids": [labels[idx] for idx in target_indices],
                        "input_ids": [labels[idx] for idx in input_indices],
                    },
                )
            )

    states, _, _, _ = forward(program)
    target_trace = [[state[idx] for idx in target_indices] for state in states]

    result = OptimalInputResult(
        input_series=[row[:] for row in program],
        activations=[state[:] for state in states],
        target_trace=[trace[:] for trace in target_trace],
        loss_history=records,
        labels=labels[:],
        metadata={
            "target_ids": [labels[idx] for idx in target_indices],
            "input_ids": [labels[idx] for idx in input_indices],
            "steps": steps,
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "divisive_norm": divisive_norm,
            "signed": signed,
        },
    )
    return result

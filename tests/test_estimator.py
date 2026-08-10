from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP

from estimator import Estimator


def _build_mlp(width: int, depth: int, seed: int = 0) -> MLP:
    rng = fnp.random.default_rng(seed)
    scale = (2.0 / width) ** 0.5
    weights = [
        fnp.array(rng.standard_normal((width, width), dtype=fnp.float32) * scale)
        for _ in range(depth)
    ]
    return MLP(width=width, depth=depth, weights=weights, seed=seed)


def _run(mlp: MLP, budget: int) -> tuple[fnp.ndarray, int, bool]:
    with flops.BudgetContext(flop_budget=budget, quiet=True) as context:
        prediction = Estimator().predict(mlp, budget)
        finite = bool(fnp.all(fnp.isfinite(prediction)))
    return prediction, int(context.flops_used), finite


def test_contract_shape_type_finite_and_nonnegative(monkeypatch) -> None:
    monkeypatch.setenv("SOL_CV32_MAX_SAMPLES", "32")
    mlp = _build_mlp(width=8, depth=4, seed=11)
    prediction, used, finite = _run(mlp, budget=1_000_000_000)

    assert isinstance(prediction, fnp.ndarray)
    assert prediction.shape == (4, 8)
    assert finite
    assert used > 0
    assert min(min(row) for row in prediction.tolist()) >= 0.0


def test_prediction_is_deterministic_for_same_mlp(monkeypatch) -> None:
    monkeypatch.setenv("SOL_CV32_MAX_SAMPLES", "32")
    mlp = _build_mlp(width=8, depth=4, seed=7)

    first, _, _ = _run(mlp, budget=1_000_000_000)
    second, _, _ = _run(mlp, budget=1_000_000_000)

    assert first.tolist() == second.tolist()


def test_depth_one_returns_exact_shape(monkeypatch) -> None:
    monkeypatch.setenv("SOL_CV32_MAX_SAMPLES", "16")
    mlp = _build_mlp(width=6, depth=1, seed=3)
    prediction, _, finite = _run(mlp, budget=100_000_000)

    assert prediction.shape == (1, 6)
    assert finite


def test_sample_count_is_bounded_and_antithetic(monkeypatch) -> None:
    monkeypatch.setenv("SOL_CV32_MAX_SAMPLES", "123")
    count = Estimator._choose_sample_count(
        width=16,
        depth=8,
        budget=100_000_000_000,
        full_covariance=True,
    )

    assert 0 <= count <= 123
    assert count % 4 == 0


def test_low_budget_can_disable_sampling(monkeypatch) -> None:
    monkeypatch.setenv("SOL_CV32_MAX_SAMPLES", "4608")
    count = Estimator._choose_sample_count(
        width=256,
        depth=32,
        budget=4_000_000_000,
        full_covariance=True,
    )

    assert count == 0

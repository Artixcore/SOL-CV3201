"""Run a contract and budget smoke test without importing NumPy."""

from __future__ import annotations

import argparse

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP

from estimator import Estimator


def build_mlp(width: int, depth: int, seed: int) -> MLP:
    rng = fnp.random.default_rng(seed)
    scale = (2.0 / width) ** 0.5
    weights = [
        fnp.array(rng.standard_normal((width, width), dtype=fnp.float32) * scale)
        for _ in range(depth)
    ]
    return MLP(width=width, depth=depth, weights=weights, seed=seed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--budget", type=int, default=50_000_000_000)
    args = parser.parse_args()

    mlp = build_mlp(args.width, args.depth, args.seed)
    with flops.BudgetContext(flop_budget=args.budget, quiet=True) as context:
        prediction = Estimator().predict(mlp, args.budget)
        finite = bool(fnp.all(fnp.isfinite(prediction)))
        nonnegative = bool(fnp.all(prediction >= 0.0))

    expected = (args.depth, args.width)
    if prediction.shape != expected:
        raise SystemExit(f"wrong shape: got {prediction.shape}, expected {expected}")
    if not finite:
        raise SystemExit("prediction contains NaN or Inf")
    if not nonnegative:
        raise SystemExit("prediction contains a negative ReLU mean")
    if context.flops_used > args.budget:
        raise SystemExit("estimator exceeded the supplied budget")

    print(
        "SOL-CV32 smoke test passed: "
        f"shape={prediction.shape}, flops={context.flops_used:,}, "
        f"budget={args.budget:,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

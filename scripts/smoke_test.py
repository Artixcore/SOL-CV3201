"""Run a contract and budget smoke test without importing NumPy."""

from __future__ import annotations

import argparse
import time

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
    parser.add_argument("--max-fraction", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=None)
    args = parser.parse_args()

    if not 0.0 < args.max_fraction <= 1.0:
        raise SystemExit("--max-fraction must be in (0, 1]")
    if args.max_seconds is not None and args.max_seconds <= 0.0:
        raise SystemExit("--max-seconds must be positive")

    mlp = build_mlp(args.width, args.depth, args.seed)
    started = time.monotonic()
    with flops.BudgetContext(flop_budget=args.budget, quiet=True) as context:
        prediction = Estimator().predict(mlp, args.budget)
        finite = bool(fnp.all(fnp.isfinite(prediction)))
        nonnegative = bool(fnp.all(prediction >= 0.0))
    elapsed = time.monotonic() - started

    expected = (args.depth, args.width)
    if prediction.shape != expected:
        raise SystemExit(f"wrong shape: got {prediction.shape}, expected {expected}")
    if not finite:
        raise SystemExit("prediction contains NaN or Inf")
    if not nonnegative:
        raise SystemExit("prediction contains a negative ReLU mean")

    flop_limit = int(args.budget * args.max_fraction)
    if context.flops_used > flop_limit:
        raise SystemExit(
            f"estimator used {context.flops_used:,} FLOPs, above limit {flop_limit:,}"
        )
    if args.max_seconds is not None and elapsed > args.max_seconds:
        raise SystemExit(
            f"estimator took {elapsed:.3f}s, above limit {args.max_seconds:.3f}s"
        )

    print(
        "SOL-CV32 smoke test passed: "
        f"shape={prediction.shape}, flops={context.flops_used:,}, "
        f"flop_limit={flop_limit:,}, elapsed={elapsed:.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

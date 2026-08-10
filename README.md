# SOL-CV32 v0.1

SOL-CV32 is an experimental estimator for the ARC White-Box Estimation Challenge 2026.

The v0.1 design combines:

- full-covariance Gaussian moment propagation as a deterministic white-box anchor,
- antithetic Gaussian sampling,
- exact first-layer ReLU means used as zero-mean control variates,
- two-fold cross-fitted ridge control-variate regression for the final layer,
- adaptive shrinkage between the analytical anchor and the control-variate estimate,
- budget-aware sample sizing with a conservative target below the competition's 10% compute-discount threshold.

This repository is a research prototype. Passing CI means the estimator satisfies the starter-kit contract and smoke tests. It does not by itself prove leaderboard competitiveness.

## Challenge contract

`Estimator.predict(mlp, budget)` returns a `flopscope.numpy.ndarray` of shape `(mlp.depth, mlp.width)` with finite expected post-ReLU activation estimates.

The implementation targets Python 3.10 and the Phase 1/Phase 2 starter-kit dependency family:

- `flopscope >=0.10.0,<0.11.0`
- `whestbench >=0.14.0,<0.15.0`

## Local validation

```bash
python -m pip install -e '.[dev]'
ruff check .
pytest -q
whest validate --estimator estimator.py
```

## CI

GitHub Actions runs linting, unit tests, the official `whest validate` contract check, and a 32-layer width-256 competition-shape smoke test with a reduced sample cap for CI runtime.

## Submission

Do not submit v0.1 blindly. Benchmark it on the official public WhestBench data first, inspect final-layer MSE and actual FLOP use, then tune sample count, ridge strength, covariance strategy, and shrinkage before packaging.

## License

MIT. See `LICENSE`.

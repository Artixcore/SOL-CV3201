# SOL-CV32 v0.1

SOL-CV32 is an experimental estimator for the ARC White-Box Estimation Challenge 2026.

The v0.1 design combines:

- Hermite-enhanced full-covariance Gaussian moment propagation as a deterministic white-box anchor,
- exact marginal ReLU means and variances,
- antithetic Gaussian sampling with paired inputs `x` and `-x`,
- exact first-layer ReLU means used as zero-mean control variates,
- two-fold cross-fitted ridge control-variate regression for the final layer,
- output-wise variance-reduction gating,
- adaptive shrinkage between the analytical anchor and sampled correction,
- budget-aware sample sizing targeting about 9.2% of the supplied budget, with a default maximum of 4,608 raw paths.

This repository is a research prototype. Passing CI means the estimator satisfies the starter-kit contract and smoke tests. It does not prove leaderboard competitiveness or private-suite performance.

## Challenge contract

`Estimator.predict(mlp, budget)` returns a `flopscope.numpy.ndarray` of shape `(mlp.depth, mlp.width)` containing finite, nonnegative expected post-ReLU activation estimates.

The implementation is self-contained at grading time and imports only `flopscope`, `flopscope.numpy`, the WhestBench API, and Python's standard library.

The project targets Python 3.10 and the current starter-kit dependency family:

- `flopscope >=0.10.0,<0.11.0`
- `whestbench >=0.14.0,<0.15.0`

Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the estimator pipeline, budget policy, limitations, and planned experiments.

## Local validation

```bash
python -m pip install -e '.[dev]'
ruff check .
pytest -q
whest validate --estimator estimator.py
SOL_CV32_MAX_SAMPLES=32 python scripts/smoke_test.py --width 256 --depth 32 --budget 50000000000
```

`SOL_CV32_MAX_SAMPLES` is a local and CI override. The submitted estimator does not require it and uses the built-in default when the variable is absent.

## CI

GitHub Actions runs:

1. Ruff linting.
2. Unit tests for shape, type, finiteness, nonnegativity, determinism, depth-one behavior, and budget planning.
3. The official `whest validate --estimator estimator.py` contract check.
4. A width-256, depth-32 smoke test with a reduced stochastic sample cap for CI runtime.
5. Packaging of `dist/SOL-CV32-v0.1.tar.gz` as a downloadable workflow artifact.

## Submission preparation

Do not submit v0.1 blindly. First run it on the official public WhestBench Mini split, compare final-layer MSE and adjusted score against mean propagation, covariance propagation, and equal-compute Monte Carlo, then tune:

- Hermite order,
- stochastic sample count,
- ridge regularization,
- covariance routing,
- correction gate,
- shrinkage calibration,
- target FLOP fraction.

To build the single-file submission locally:

```bash
mkdir -p dist
whest package --estimator estimator.py --output dist/SOL-CV32-v0.1.tar.gz
```

## License

MIT. See [LICENSE](LICENSE).

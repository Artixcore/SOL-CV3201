# SOL-CV32 v0.1 architecture

## Objective

For each hidden layer of a random ReLU MLP, estimate the activation mean under a standard Gaussian input. The primary challenge score uses final-layer mean squared error and analytically counted FLOPs.

## Estimation pipeline

### 1. Hermite-enhanced covariance anchor

The deterministic anchor carries a mean vector and full covariance matrix. Linear propagation is exact under the represented moments:

```text
mu_pre  = W.T @ mu
cov_pre = W.T @ cov @ W
```

For each marginal Gaussian preactivation, the ReLU mean and variance are exact. Off-diagonal covariance uses the first three terms of the Hermite correlation series. The first term is the standard gain approximation; the second and third terms retain quadratic and cubic correlation effects. The diagonal is then replaced by the exact marginal ReLU variance.

When a full covariance pass is unsafe for the supplied budget, the estimator falls back to diagonal mean/variance propagation.

### 2. Antithetic paths

The stochastic component evaluates paired inputs `x` and `-x`. Pair averaging cancels odd input components and typically lowers variance compared with independent paths at the same number of forward passes.

### 3. Exact first-layer controls

The first preactivation is exactly Gaussian because the input is Gaussian and the first operation is linear. Therefore each first-layer ReLU mean is available analytically. The centered first-layer pair activations have known expectation zero and are valid control variates.

### 4. Cross-fitted ridge correction

The independent antithetic pairs are split into two folds. A centered multi-output ridge map is fitted on each fold and applied only to the opposite fold. This separation reduces the bias risk that would arise from fitting and correcting the same observations.

The correction is gated per output by observed held-out variance reduction. Outputs for which the control does not reduce variance remain close to raw antithetic sampling.

### 5. Analytical shrinkage

Sample corrections are shrunk toward the white-box anchor only when the observed anchor-to-sample disagreement is not clearly larger than the estimated sampling error. This is an empirical calibration rule, not a theorem, and must be tuned on the public benchmark.

## Budget policy

The default planner targets about 9.2% of the supplied budget, below the scoring rule's 10% compute-discount boundary. It reserves estimated costs for covariance propagation, two ridge solves, and bookkeeping before choosing a multiple-of-four sample count. The default maximum is 4,608 raw paths.

Set `SOL_CV32_MAX_SAMPLES` locally or in CI to cap the stochastic component. The grader needs no environment variable; the default is built in.

## Known limitations

- v0.1 has not been calibrated on the official private suite.
- Gaussian moment propagation accumulates approximation error with depth.
- First-layer controls may become weak after many random layers.
- The empirical shrinkage and variance gate can help on average while hurting individual networks.
- Passing CI proves contract correctness, not leaderboard quality.

## Planned experiments

1. Compare Hermite orders 1 through 6.
2. Tune ridge strength and control rank on the public Mini split.
3. Test low-rank randomized controls and sensitivity-guided controls.
4. Compare full covariance for all layers with early-layer-only covariance routing.
5. Tune the target budget fraction using actual `flopscope` reports.
6. Run ablations for antithetic pairing, control variates, and analytical shrinkage.

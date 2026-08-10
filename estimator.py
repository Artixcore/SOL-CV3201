"""SOL-CV32 v0.1: structured white-box plus control-variate estimation.

The estimator is intentionally self-contained so the single ``estimator.py``
file can be packaged for the ARC White-Box Estimation Challenge.

Pipeline
--------
1. Propagate Gaussian means and a full covariance matrix through every ReLU
   layer. This is the deterministic white-box anchor.
2. Draw antithetic Gaussian inputs ``x`` and ``-x`` and run them through the
   supplied MLP.
3. Use the exact first-layer ReLU mean as a zero-mean control variate.
4. Fit two ridge regressions on disjoint antithetic-pair folds and apply each
   regression only to the opposite fold (cross-fitting).
5. Shrink noisy sample corrections toward the analytical anchor using an
   empirical signal-to-noise estimate.

Only ``flopscope``, ``flopscope.numpy`` and the WhestBench API are imported.
No NumPy, SciPy, network access or external model is needed at grading time.
"""

from __future__ import annotations

import os

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, MLP

_VERSION = "0.1.0"

# The public scoring multiplier bottoms out at 0.1 when C/B <= 0.1. We aim a
# little below that boundary to leave room for cost-model drift and small
# bookkeeping operations.
_TARGET_BUDGET_FRACTION = 0.092

# 4608 raw paths means 2304 independent antithetic pairs. At width=256 and
# depth=32 this normally keeps the hybrid comfortably below the 10% budget
# frontier once the covariance anchor and cross-fit solve are included.
_DEFAULT_MAX_SAMPLES = 4608
_ABSOLUTE_MAX_SAMPLES = 8192
_MIN_SAMPLES = 8

_RIDGE_RELATIVE = 1.0
_RIDGE_ABSOLUTE = 1.0e-6
_BLEND_MAX = 1.0
_EPS = 1.0e-12

# Float32 overflow prevention for unusually unstable networks. ReLU is
# positively homogeneous, so rescaling mean/covariance and restoring the
# accumulated scale in recorded means preserves the propagated estimate.
_COV_RESCALE_THRESHOLD = 1.0e20


class Estimator(BaseEstimator):
    """Budget-aware SOL-CV32 v0.1 estimator."""

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        """Return expected post-ReLU activations with shape ``(depth, width)``."""
        width = int(mlp.width)
        depth = int(mlp.depth)
        budget_i = max(int(budget), 1)

        full_covariance = self._should_use_full_covariance(width, depth, budget_i)
        if full_covariance:
            anchor = self._covariance_anchor(mlp)
        else:
            anchor = self._diagonal_anchor(mlp)

        sample_count = self._choose_sample_count(
            width=width,
            depth=depth,
            budget=budget_i,
            full_covariance=full_covariance,
        )
        if sample_count < _MIN_SAMPLES:
            return self._sanitize(anchor)

        raw_means, raw_se2, controls, final_pairs = self._antithetic_pass(
            mlp=mlp,
            sample_count=sample_count,
            exact_first_mean=anchor[0],
        )

        # Use low-variance antithetic means to gently correct intermediate
        # layers. Layer zero remains the exact analytical first-layer mean.
        rows = [anchor[0]]
        for layer_index in range(1, depth - 1):
            rows.append(
                self._shrink_to_anchor(
                    anchor[layer_index],
                    raw_means[layer_index],
                    raw_se2[layer_index],
                )
            )

        final_cv_mean, final_cv_se2 = self._cross_fitted_control_variate(
            controls=controls,
            targets=final_pairs,
        )
        final_row = self._shrink_to_anchor(anchor[-1], final_cv_mean, final_cv_se2)

        if depth > 1:
            rows.append(final_row)
        else:
            # A depth-one network has an exact analytical first-layer mean.
            rows = [anchor[0]]

        return self._sanitize(fnp.stack(rows, axis=0))

    @staticmethod
    def _should_use_full_covariance(width: int, depth: int, budget: int) -> bool:
        # The symmetric covariance sandwich is approximately 6.5*n^3 per
        # layer under the current flopscope model, including ReLU bookkeeping.
        estimated = int(6.5 * depth * width * width * width + 16 * depth * width * width)
        return estimated <= int(0.82 * budget)

    @staticmethod
    def _read_sample_cap() -> int:
        """Read an optional local/CI cap without making it a grader dependency."""
        raw = os.getenv("SOL_CV32_MAX_SAMPLES")
        if raw is None:
            return _DEFAULT_MAX_SAMPLES
        try:
            value = int(raw)
        except ValueError:
            return _DEFAULT_MAX_SAMPLES
        return max(0, min(value, _ABSOLUTE_MAX_SAMPLES))

    @classmethod
    def _choose_sample_count(
        cls,
        *,
        width: int,
        depth: int,
        budget: int,
        full_covariance: bool,
    ) -> int:
        """Choose a conservative multiple-of-four antithetic path count."""
        max_samples = cls._read_sample_cap()
        if max_samples < _MIN_SAMPLES:
            return 0

        if full_covariance:
            anchor_cost = int(
                6.5 * depth * width * width * width + 16 * depth * width * width
            )
        else:
            anchor_cost = int(8 * depth * width * width + 24 * depth * width)

        # Two width-by-width ridge solves plus conservative matrix overhead.
        regression_fixed = int(7 * width * width * width + 64 * width * width)

        # A raw path costs roughly one width-by-width matmul per layer. The
        # extra 4*n^2 term reserves control-variates, means and centering.
        per_sample_cost = int((2 * depth + 4) * width * width + 8 * depth * width)

        target = int(_TARGET_BUDGET_FRACTION * budget)
        reserve = anchor_cost + regression_fixed + max(1_000_000, int(0.001 * budget))
        available = target - reserve
        if available <= 0 or per_sample_cost <= 0:
            return 0

        sample_count = min(max_samples, int(available // per_sample_cost))
        sample_count -= sample_count % 4
        return sample_count if sample_count >= _MIN_SAMPLES else 0

    @staticmethod
    def _normal_pdf_cdf(alpha: fnp.ndarray) -> tuple[fnp.ndarray, fnp.ndarray]:
        # Explicit float32 casts keep downstream covariance/sampling work on
        # the intended cost path even if a backend returns float64 statistics.
        pdf = fnp.asarray(flops.stats.norm.pdf(alpha), dtype=fnp.float32)
        cdf = fnp.asarray(flops.stats.norm.cdf(alpha), dtype=fnp.float32)
        return pdf, cdf

    def _diagonal_anchor(self, mlp: MLP) -> fnp.ndarray:
        """Fast fallback: propagate independent Gaussian means/variances."""
        width = int(mlp.width)
        mu = fnp.zeros(width, dtype=fnp.float32)
        var = fnp.ones(width, dtype=fnp.float32)
        rows = []

        for weight in mlp.weights:
            mu_pre = weight.T @ mu
            var_pre = fnp.maximum((weight * weight).T @ var, _EPS)
            sigma_pre = fnp.sqrt(var_pre)
            alpha = mu_pre / sigma_pre
            pdf, cdf = self._normal_pdf_cdf(alpha)

            mu = fnp.asarray(mu_pre * cdf + sigma_pre * pdf, dtype=fnp.float32)
            second = (mu_pre * mu_pre + var_pre) * cdf + mu_pre * sigma_pre * pdf
            var = fnp.asarray(fnp.maximum(second - mu * mu, 0.0), dtype=fnp.float32)
            rows.append(mu)

        return fnp.stack(rows, axis=0)

    def _covariance_anchor(self, mlp: MLP) -> fnp.ndarray:
        """Propagate a full Gaussian covariance through all ReLU layers."""
        width = int(mlp.width)
        mu = fnp.zeros(width, dtype=fnp.float32)
        cov = flops.as_symmetric(
            fnp.eye(width, dtype=fnp.float32),
            symmetry=(0, 1),
        )
        log_scale = 0.0
        rows = []

        for weight in mlp.weights:
            cov_diag = fnp.diag(cov)
            max_var = float(fnp.max(cov_diag))
            if max_var > _COV_RESCALE_THRESHOLD:
                scale = float(fnp.sqrt(max_var))
                mu = fnp.asarray(mu / scale, dtype=fnp.float32)
                cov = fnp.asarray(cov / (scale * scale), dtype=fnp.float32)
                cov = flops.as_symmetric(cov, symmetry=(0, 1))
                log_scale += float(fnp.log(scale))

            mu_pre = weight.T @ mu
            cov_pre = fnp.einsum("ij,ia,jb->ab", cov, weight, weight)
            var_pre = fnp.maximum(fnp.diag(cov_pre), _EPS)
            sigma_pre = fnp.sqrt(var_pre)
            alpha = mu_pre / sigma_pre
            pdf, cdf = self._normal_pdf_cdf(alpha)

            mu = fnp.asarray(mu_pre * cdf + sigma_pre * pdf, dtype=fnp.float32)
            second = (mu_pre * mu_pre + var_pre) * cdf + mu_pre * sigma_pre * pdf
            var_post = fnp.asarray(
                fnp.maximum(second - mu * mu, 0.0),
                dtype=fnp.float32,
            )

            # Hermite-enhanced off-diagonal covariance. The first term is
            # the usual gain approximation. Terms two and three retain the
            # next powers of the Gaussian correlation without material cubic
            # cost; the exact marginal variance still replaces the diagonal.
            denom = fnp.maximum(fnp.outer(sigma_pre, sigma_pre), _EPS)
            rho = cov_pre / denom
            rho = fnp.minimum(fnp.maximum(rho, -0.999), 0.999)

            b1 = sigma_pre * cdf
            cov = fnp.outer(b1, b1) * rho

            rho_power = rho * rho
            b2 = sigma_pre * pdf
            cov = cov + fnp.outer(b2, b2) * rho_power * 0.5

            rho_power = rho_power * rho
            b3 = -sigma_pre * alpha * pdf
            cov = cov + fnp.outer(b3, b3) * rho_power / 6.0

            fnp.fill_diagonal(cov, var_post)
            cov = flops.as_symmetric(cov, symmetry=(0, 1))

            scale_factor = fnp.asarray(fnp.exp(log_scale), dtype=fnp.float32)
            rows.append(fnp.asarray(mu * scale_factor, dtype=fnp.float32))

        return fnp.stack(rows, axis=0)

    def _antithetic_pass(
        self,
        *,
        mlp: MLP,
        sample_count: int,
        exact_first_mean: fnp.ndarray,
    ) -> tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray]:
        """Run antithetic paths and return layer statistics plus CV data."""
        pair_count = sample_count // 2
        seed = int(getattr(mlp, "seed", 0) or 0) ^ 0x5A17C032
        rng = fnp.random.default_rng(seed)
        base = fnp.array(
            rng.standard_normal((pair_count, mlp.width), dtype=fnp.float32)
        )
        activations = fnp.concatenate((base, -base), axis=0)

        layer_means = []
        layer_se2 = []
        controls = None
        final_pairs = None

        for layer_index, weight in enumerate(mlp.weights):
            activations = fnp.maximum(activations @ weight, 0.0)
            paired = (activations[:pair_count] + activations[pair_count:]) * 0.5
            mean, se2 = self._mean_and_se2(paired)
            layer_means.append(mean)
            layer_se2.append(se2)

            if layer_index == 0:
                controls = fnp.asarray(paired - exact_first_mean, dtype=fnp.float32)
            if layer_index == mlp.depth - 1:
                final_pairs = fnp.asarray(paired, dtype=fnp.float32)

        # These are guaranteed by width/depth >= 1 in the challenge contract.
        assert controls is not None
        assert final_pairs is not None
        return (
            fnp.stack(layer_means, axis=0),
            fnp.stack(layer_se2, axis=0),
            controls,
            final_pairs,
        )

    @staticmethod
    def _mean_and_se2(values: fnp.ndarray) -> tuple[fnp.ndarray, fnp.ndarray]:
        count = max(int(values.shape[0]), 1)
        mean = fnp.mean(values, axis=0)
        centered = values - mean
        variance = fnp.mean(centered * centered, axis=0)
        se2 = fnp.maximum(variance / float(count), _EPS)
        return (
            fnp.asarray(mean, dtype=fnp.float32),
            fnp.asarray(se2, dtype=fnp.float32),
        )

    def _fit_ridge(self, controls: fnp.ndarray, targets: fnp.ndarray) -> fnp.ndarray:
        """Fit a centered multi-output ridge map controls -> targets."""
        count = max(int(controls.shape[0]), 1)
        c_centered = controls - fnp.mean(controls, axis=0)
        y_centered = targets - fnp.mean(targets, axis=0)
        inv_count = 1.0 / float(count)

        gram = fnp.asarray(c_centered.T @ c_centered * inv_count, dtype=fnp.float32)
        cross = fnp.asarray(c_centered.T @ y_centered * inv_count, dtype=fnp.float32)
        diagonal_scale = fnp.mean(fnp.diag(gram))
        ridge = fnp.maximum(diagonal_scale * _RIDGE_RELATIVE, _RIDGE_ABSOLUTE)
        system = gram + fnp.eye(gram.shape[0], dtype=fnp.float32) * ridge
        return fnp.asarray(fnp.linalg.solve(system, cross), dtype=fnp.float32)

    def _cross_fitted_control_variate(
        self,
        *,
        controls: fnp.ndarray,
        targets: fnp.ndarray,
    ) -> tuple[fnp.ndarray, fnp.ndarray]:
        """Estimate the final mean with opposite-fold ridge corrections."""
        pair_count = int(controls.shape[0])
        split = pair_count // 2
        if split < 2 or pair_count - split < 2:
            return self._mean_and_se2(targets)

        controls_a = controls[:split]
        controls_b = controls[split:]
        targets_a = targets[:split]
        targets_b = targets[split:]

        beta_a = self._fit_ridge(controls_a, targets_a)
        beta_b = self._fit_ridge(controls_b, targets_b)

        corrected_a = targets_a - controls_a @ beta_b
        corrected_b = targets_b - controls_b @ beta_a
        corrected = fnp.concatenate((corrected_a, corrected_b), axis=0)

        # Cross-fitted regression is not guaranteed to help every output,
        # especially after 32 random layers. Gate it by observed held-out
        # variance reduction, then measure uncertainty on the gated samples.
        raw_mean = fnp.mean(targets, axis=0)
        corrected_mean = fnp.mean(corrected, axis=0)
        raw_centered = targets - raw_mean
        corrected_centered = corrected - corrected_mean
        raw_variance = fnp.mean(raw_centered * raw_centered, axis=0)
        corrected_variance = fnp.mean(corrected_centered * corrected_centered, axis=0)
        gate = fnp.maximum(raw_variance - corrected_variance, 0.0) / (raw_variance + _EPS)
        gate = fnp.minimum(fnp.maximum(gate, 0.0), 1.0)
        gated = targets + (corrected - targets) * gate
        return self._mean_and_se2(gated)

    @staticmethod
    def _shrink_to_anchor(
        anchor: fnp.ndarray,
        sampled_mean: fnp.ndarray,
        sampled_se2: fnp.ndarray,
    ) -> fnp.ndarray:
        """Empirical-Bayes-style shrinkage of a noisy correction."""
        delta = sampled_mean - anchor
        estimated_signal = fnp.maximum(delta * delta - sampled_se2, 0.0)
        weight = estimated_signal / (estimated_signal + sampled_se2 + _EPS)
        weight = fnp.minimum(fnp.maximum(weight, 0.0), _BLEND_MAX)
        return fnp.asarray(anchor + weight * delta, dtype=fnp.float32)

    @staticmethod
    def _sanitize(values: fnp.ndarray) -> fnp.ndarray:
        finite = fnp.where(fnp.isfinite(values), values, 0.0)
        return fnp.asarray(fnp.maximum(finite, 0.0), dtype=fnp.float32)


__all__ = ["Estimator"]

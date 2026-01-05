from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, Any, Dict, List, Optional, Tuple
import math
import numpy as np


Executor = Callable[[Any], float]


@dataclass(frozen=True)
class ZNEResult:
    zne: float
    ci_low: float
    ci_high: float
    stderr: float
    # Raw per-scale statistics for debugging/plots
    scales: Tuple[float, ...]
    means: Tuple[float, ...]
    variances: Tuple[float, ...]
    n_repeats: int
    degree: int


@dataclass(frozen=True)
class ZNEBlowUp:
    circuit_id: int
    # Errors vs ideal
    noisy_err: float
    zne_err: float
    # Variance inflation diagnostics
    zne_ci_width: float
    # Optional metadata to help you stratify the failure mode
    depth: int
    n_cx: int
    n_1q: int
    reason: str


def _z_for_two_sided_ci(ci_level: float) -> float:
    """
    Returns z for a two-sided normal CI (approx).
    Common: 0.95 -> 1.96, 0.99 -> 2.576.
    """
    if not (0.0 < ci_level < 1.0):
        raise ValueError("ci_level must be in (0, 1)")
    # Minimal mapping to avoid scipy dependency
    if abs(ci_level - 0.95) < 1e-9:
        return 1.959963984540054
    if abs(ci_level - 0.99) < 1e-9:
        return 2.5758293035489004
    # Fallback: conservative normal approx using inverse error function
    # z = sqrt(2) * erfinv(ci_level)
    # Two-sided: alpha = 1 - ci_level, use (1 - alpha/2)
    p = 1.0 - (1.0 - ci_level) / 2.0
    return math.sqrt(2.0) * _erfinv(2.0 * p - 1.0)


def _erfinv(x: float) -> float:
    """
    Approx inverse error function (good enough for CI z-values).
    Reference: Winitzki approximation.
    """
    # Clamp for numeric safety
    x = max(-0.999999, min(0.999999, x))
    a = 0.147
    ln = math.log(1.0 - x * x)
    t = 2.0 / (math.pi * a) + ln / 2.0
    return math.copysign(math.sqrt(math.sqrt(t * t - ln / a) - t), x)


def _repeat_execute(executor: Executor, circuit: Any, n_repeats: int) -> np.ndarray:
    """
    Executes the same circuit multiple times to empirically capture stochasticity:
    shot noise, Aer sampling randomness, and any executor nondeterminism.
    """
    vals = np.empty(n_repeats, dtype=np.float64)
    for i in range(n_repeats):
        vals[i] = float(executor(circuit))
    return vals


def _wls_extrapolate_to_zero(
    scales: np.ndarray,
    means: np.ndarray,
    variances_of_means: np.ndarray,
    degree: int,
) -> Tuple[float, float]:
    """
    Weighted least squares polynomial fit y(s) = b0 + b1*s + b2*s^2 + ...
    Returns (b0, Var(b0)) where b0 is the extrapolated value at s=0.

    variances_of_means are the variances of the *estimated means* at each scale,
    so weights are w_i = 1 / var_i.
    """
    if degree < 1:
        raise ValueError("degree must be >= 1")
    if len(scales) != len(means) or len(scales) != len(variances_of_means):
        raise ValueError("scales/means/variances length mismatch")
    if len(scales) < degree + 1:
        raise ValueError("Need at least degree+1 scale points")

    # Design matrix: [1, s, s^2, ...]
    X = np.vstack([scales ** k for k in range(degree + 1)]).T  # (n, p)
    # Guard against zeros
    v = np.maximum(variances_of_means, 1e-18)
    W = np.diag(1.0 / v)

    XtW = X.T @ W
    XtWX = XtW @ X
    try:
        cov = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        # Singular: fall back to pseudo-inverse (still yields a covariance-like matrix)
        cov = np.linalg.pinv(XtWX)

    beta = cov @ (XtW @ means)
    b0 = float(beta[0])
    var_b0 = float(cov[0, 0])
    var_b0 = max(var_b0, 0.0)
    return b0, var_b0


def zne_with_ci(
    executor: Executor,
    circuit: Any,
    scale_noise: Callable[[Any, float], Any],
    scale_factors: Sequence[float] = (1.0, 2.0, 3.0),
    n_repeats: int = 25,
    degree: Optional[int] = None,
    ci_level: float = 0.95,
    rng_seed: Optional[int] = None,
) -> ZNEResult:
    """
    CI-aware ZNE using repeated execution at each scale factor and WLS covariance propagation.

    - Runs the folded circuit n_repeats times per scale.
    - Computes mean and variance of the mean at each scale.
    - Fits a polynomial in the scale factor using WLS.
    - Returns b0 with a normal-approx CI.

    Notes:
    - This is intentionally independent of Mitiq so you can validate correctness.
    - It matches the executor signature used in this repo: executor(qc)->float
      (see NoisyExecutorPool in eval_benchmarks.py) :contentReference[oaicite:2]{index=2}.
    """
    if rng_seed is not None:
        np.random.seed(int(rng_seed))

    sf = tuple(float(s) for s in scale_factors)
    if 1.0 not in sf:
        raise ValueError("scale_factors must include 1.0")
    if len(sf) < 3:
        raise ValueError("Use at least 3 scale factors for stable CI diagnostics")
    if n_repeats < 5:
        raise ValueError("n_repeats should be >= 5 to estimate variance")

    deg = degree
    if deg is None:
        # Conservative default: linear unless you have many points
        deg = 1 if len(sf) < 4 else 2
    deg = int(deg)

    scales = np.array(sf, dtype=np.float64)

    per_scale_means = []
    per_scale_vars = []
    for s in scales:
        c_folded = circuit if abs(s - 1.0) < 1e-12 else scale_noise(circuit, float(s))
        samples = _repeat_execute(executor, c_folded, n_repeats=n_repeats)
        m = float(np.mean(samples))
        # unbiased sample variance of the raw samples
        raw_var = float(np.var(samples, ddof=1))
        # variance of the mean
        var_mean = raw_var / float(n_repeats)
        per_scale_means.append(m)
        per_scale_vars.append(var_mean)

    means = np.array(per_scale_means, dtype=np.float64)
    var_means = np.array(per_scale_vars, dtype=np.float64)

    b0, var_b0 = _wls_extrapolate_to_zero(scales, means, var_means, degree=deg)

    z = _z_for_two_sided_ci(ci_level)
    stderr = math.sqrt(var_b0)
    ci_low = b0 - z * stderr
    ci_high = b0 + z * stderr

    return ZNEResult(
        zne=float(b0),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        stderr=float(stderr),
        scales=tuple(scales.tolist()),
        means=tuple(means.tolist()),
        variances=tuple(var_means.tolist()),
        n_repeats=int(n_repeats),
        degree=int(deg),
    )


def _circuit_stats(qc: Any) -> Tuple[int, int, int]:
    """
    Extracts simple circuit complexity proxies: depth, #CX, #1q.
    Works for Qiskit QuantumCircuit; falls back if unavailable.
    """
    try:
        depth = int(qc.depth())
    except Exception:
        depth = -1

    n_cx = 0
    n_1q = 0
    try:
        for inst, qargs, _cargs in qc.data:
            name = getattr(inst, "name", "")
            if name in ("cx", "cz", "swap"):
                n_cx += 1
            elif name and name not in ("measure", "barrier"):
                # approximate 1q gate count
                if len(qargs) == 1:
                    n_1q += 1
    except Exception:
        pass

    return depth, n_cx, n_1q


def detect_zne_blowups(
    circuit_ids: Sequence[int],
    circuits: Sequence[Any],
    ideals: Sequence[float],
    noisies: Sequence[float],
    zne_results: Sequence[ZNEResult],
    *,
    err_ratio_thresh: float = 2.0,
    abs_err_thresh: float = 0.25,
    ci_width_thresh: float = 0.25,
) -> List[ZNEBlowUp]:
    """
    Flags circuits where ZNE is clearly unstable.

    A circuit is flagged if any of:
    - |zne_err| >= err_ratio_thresh * |noisy_err|  (catastrophic relative worsening)
    - |zne_err| >= abs_err_thresh                  (absolute failure)
    - CI width >= ci_width_thresh                  (variance inflation)

    Tune thresholds to your observable range; for Pauli expectations in [-1,1],
    ci_width_thresh in 0.2-0.4 is usually informative.
    """
    blowups: List[ZNEBlowUp] = []
    for cid, qc, y_ideal, y_noisy, zne_res in zip(circuit_ids, circuits, ideals, noisies, zne_results):
        noisy_err = float(y_noisy - y_ideal)
        zne_err = float(zne_res.zne - y_ideal)
        noisy_abs = abs(noisy_err)
        zne_abs = abs(zne_err)
        ci_width = float(zne_res.ci_high - zne_res.ci_low)

        reasons = []
        if noisy_abs > 1e-12 and zne_abs >= err_ratio_thresh * noisy_abs:
            reasons.append(f"err_ratio>={err_ratio_thresh:g}")
        if zne_abs >= abs_err_thresh:
            reasons.append(f"abs_err>={abs_err_thresh:g}")
        if ci_width >= ci_width_thresh:
            reasons.append(f"ci_width>={ci_width_thresh:g}")

        if reasons:
            depth, n_cx, n_1q = _circuit_stats(qc)
            blowups.append(
                ZNEBlowUp(
                    circuit_id=int(cid),
                    noisy_err=float(noisy_abs),
                    zne_err=float(zne_abs),
                    zne_ci_width=float(ci_width),
                    depth=int(depth),
                    n_cx=int(n_cx),
                    n_1q=int(n_1q),
                    reason=";".join(reasons),
                )
            )

    # Sort: worst first by zne_err, then CI width
    blowups.sort(key=lambda b: (b.zne_err, b.zne_ci_width), reverse=True)
    return blowups

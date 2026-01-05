from __future__ import annotations
from typing import Callable, Sequence, Optional
import inspect

try:
    from mitiq import zne
    from mitiq.zne.scaling import folding
except Exception as e:
    raise RuntimeError("This script requires `mitiq` (tested with 0.48.1).") from e


def mitigate_with_zne(
    executor: Callable[[object], float],
    circuit: object,
    observable=None,
    scale_factors: Sequence[float] = (1.0, 2.0, 3.0),
    extrapolator: Optional[object] = None,
    folding_method: Callable[[object, float], object] = folding.fold_global,
) -> float:
    """Run Zero-Noise Extrapolation (ZNE) with Mitiq.

    Args:
        executor: Callable that takes a *circuit-like* object and returns an expectation value (float).
                  This can internally use Qiskit Estimator or Aer backends. The signature is `executor(circuit) -> float`.
        circuit:  The reference circuit (noisier at scale=1.0).
        scale_factors: Noise scale factors. Include 1.0 and at least two >1 factors.
        extrapolator: Optional Mitiq factory/extrapolator override. If None, a polynomial factory is used.
        folding_method: A folding function mapping (circuit, scale) -> folded_circuit. Defaults to global folding.

    Returns:
        Mitigated expectation value (float).
    """
    factory = None
    if hasattr(zne, "PolyFactory") or hasattr(zne, "RichardsonFactory"):
        if extrapolator is not None and hasattr(extrapolator, "run"):
            factory = extrapolator
        else:
            deg = max(1, min(len(scale_factors) - 1, 2))
            if hasattr(zne, "PolyFactory"):
                factory = zne.PolyFactory(scale_factors=tuple(scale_factors), order=deg)
            else:
                factory = zne.RichardsonFactory(scale_factors=tuple(scale_factors))
    elif hasattr(zne, "ZNEFactory"):
        if extrapolator is None:
            try:
                from mitiq.zne.inference import PolynomialExtrapolator

                deg = max(1, min(len(scale_factors) - 1, 2))
                extrapolator = PolynomialExtrapolator(degree=deg)
            except Exception:
                extrapolator = None
        factory = zne.ZNEFactory(scale_factors=tuple(scale_factors), extrapolator=extrapolator)

    kwargs = {"factory": factory}
    params = inspect.signature(zne.execute_with_zne).parameters
    # Our executors return float expectations directly, so do not pass
    # observable to avoid Mitiq enforcing measurement-based executors.
    if "scale_noise" in params:
        kwargs["scale_noise"] = folding_method
    elif "folding" in params:
        kwargs["folding"] = folding_method
    return zne.execute_with_zne(circuit=circuit, executor=executor, **kwargs)


def _example_qiskit_executor(observable=None):
    """Example factory that returns an executor(circuit)->float using Qiskit Aer Estimator.
    Replace with your project-specific Estimator wrapper.
    """
    from qiskit_aer.primitives import Estimator
    from qiskit.quantum_info import SparsePauliOp

    est = Estimator()
    obs = observable or SparsePauliOp.from_list([("Z", 1.0)])

    def _exec(qc):
        res = est.run(qc, obs).result()
        return float(res.values[0])
    return _exec


def _demo():
    from qiskit import QuantumCircuit
    # A tiny demo circuit
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.rx(0.2, 0)
    #qc.cz(0, 0) if False else None  # placeholder to show a 2q gate could appear
    qc.h(0)

    executor = _example_qiskit_executor()
    mitigated = mitigate_with_zne(executor, qc, scale_factors=(1.0, 2.0, 3.0))
    print("ZNE mitigated value:", mitigated)


if __name__ == "__main__":
    _demo()

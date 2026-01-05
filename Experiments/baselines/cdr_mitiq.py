from __future__ import annotations
from typing import Callable, Sequence, Optional
import numpy as np

try:
    from mitiq import cdr
except Exception as e:
    raise RuntimeError("This script requires `mitiq` (tested with 0.48.1).") from e

import warnings
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendEstimator

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="The class ``qiskit.primitives.backend_estimator.BackendEstimator`` is deprecated",
)


def mitigate_with_cdr(
    executor: Callable[[object], float],
    circuit: object,
    observable=None,
    num_training_circuits: int = 30,
    generator: Optional[Callable[[object, int], Sequence[object]]] = None,
    simulator: Optional[Callable[[object], float]] = None,
) -> float:
    """Run Clifford Data Regression (CDR) with Mitiq.

    Args:
        executor: Callable that maps a circuit to an expectation value (float).
        circuit:  The target circuit to mitigate.
        observable: Observable used for expectation evaluation. Required if a default simulator is constructed.
        num_training_circuits: Number of training circuits to generate.
        generator: Optional training-set generator. If None, uses Mitiq's default generator.
        simulator: Callable producing near-ideal expectations. If None, uses an Aer noiseless backend.

    Returns:
        Mitigated expectation value.
    """
    # Use default training circuit generator if not provided
    if generator is None:
        # This uses mitiq.cdr to create near-Clifford training set
        def generator(circ, n):
            return cdr.generate_training_circuits(circ, n)
    if simulator is None:
        if observable is None:
            raise ValueError("CDR mitigation requires `observable` when constructing the default simulator.")

        backend = AerSimulator()
        estimator = BackendEstimator(backend)

        def simulator(qc) -> float:
            compiled = transpile(qc, backend)
            result = estimator.run(
                circuits=compiled,
                observables=observable,
                shots=None,
            ).result()
            return float(result.values[0])
        simulator.__annotations__["return"] = float
    return cdr.execute_with_cdr(
        circuit=circuit,
        executor=executor,
        num_training_circuits=num_training_circuits,
        generator=generator,
        simulator=simulator,
    )


def _example_qiskit_executor(observable=None):
    from qiskit_aer.primitives import Estimator
    from qiskit.quantum_info import SparsePauliOp
    est = Estimator()
    obs = observable or SparsePauliOp.from_list([("Z", 1.0)])

    def _exec(qc):
        return float(est.run(qc, obs).result().values[0])
    return _exec


def _demo():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(1)
    qc.h(0); qc.rz(0.17, 0); qc.h(0)

    executor = _example_qiskit_executor()
    mitigated = mitigate_with_cdr(executor, qc, num_training_circuits=20)
    print("CDR mitigated value:", mitigated)


if __name__ == "__main__":
    _demo()

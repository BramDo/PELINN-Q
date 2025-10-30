# pelinn/data/qiskit_dataset.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Dict, Tuple
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendEstimator
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error

@dataclass
class Sample:
    x: np.ndarray          # feature vector
    y_noisy: float
    y_ideal: float
    meta: Dict             # optional: circuit, noise params, etc.

def circuit_features(qc: QuantumCircuit) -> np.ndarray:
    counts = qc.count_ops()
    depth = qc.depth()
    n = qc.num_qubits
    n1 = sum(counts.get(g, 0) for g in ["x","y","z","h","sx","rz","ry","rx"])
    n2 = sum(counts.get(g, 0) for g in ["cx","cz","iswap","ecr"])
    return np.array([n, depth, n1, n2], dtype=np.float32)




from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error

def make_noise_model(p1_depol=0.001, p2_depol=0.01, p_amp=0.001,
                     readout_p01=0.02, readout_p10=0.02) -> NoiseModel:
    nm = NoiseModel()

    oneq_gates = ["id","x","y","z","h","sx","rx","ry","rz"]
    twoq_gates = ["cx","cz","iswap","ecr"]

    # Build quantum errors
    dep_1q = depolarizing_error(p1_depol, 1) if p1_depol else None
    dep_2q = depolarizing_error(p2_depol, 2) if p2_depol else None
    amp_1q = amplitude_damping_error(p_amp) if p_amp else None

    # Compose 1q errors (order is a modeling choice; pick one and keep it consistent)
    if dep_1q and amp_1q:
        err_1q = dep_1q.compose(amp_1q)   # or amp_1q.compose(dep_1q)
    else:
        err_1q = dep_1q or amp_1q

    if err_1q:
        nm.add_all_qubit_quantum_error(err_1q, oneq_gates)

    if dep_2q:
        nm.add_all_qubit_quantum_error(dep_2q, twoq_gates)

    # Readout error
    nm.add_all_qubit_readout_error([[1-readout_p10, readout_p10],
                                    [readout_p01,   1-readout_p01]])
    return nm

def estimate_expectation(qc: QuantumCircuit, obs: SparsePauliOp, estimator: BackendEstimator, shots: int|None) -> Tuple[float,float]:
    res = estimator.run(circuits=qc, observables=obs, parameter_values=None, shots=shots).result()
    ev = float(res.values[0])
    # variance is available via metadata; fallback compute from samples if needed
    var = float(res.metadata[0].get("variance", 0.0))
    return ev, var

def synthesize_samples(
    circuits: Sequence[QuantumCircuit],
    observables: Sequence[SparsePauliOp],
    noise_grid: Sequence[Dict],
    shots_noisy: int = 4000,
    shots_ideal: int = 0,   # 0 or None => exact
) -> Sequence[Sample]:
    """Return paired (noisy, ideal) samples."""
    backend_ideal = AerSimulator()
    backend_noisy_cache = {}

    est_ideal = BackendEstimator(backend_ideal)
    samples = []
    for circ_idx, (qc, obs) in enumerate(zip(circuits, observables)):
        feats = circuit_features(qc)
        # Ideal label
        y_star, _ = estimate_expectation(transpile(qc, backend_ideal), obs, est_ideal, shots=None if shots_ideal in (0, None) else shots_ideal)

        for ncfg in noise_grid:
            key = tuple(sorted(ncfg.items()))
            if key not in backend_noisy_cache:
                nm = make_noise_model(**ncfg)
                backend = AerSimulator(noise_model=nm)
                backend_noisy_cache[key] = (backend, BackendEstimator(backend))
            b, est_noisy = backend_noisy_cache[key]

            qc_noisy = transpile(qc, b)
            y_tilde, var = estimate_expectation(qc_noisy, obs, est_noisy, shots=shots_noisy)

            # assemble feature vector (concat circuit feats, noise params, scalar stats)
            noise_vec = np.array([ncfg.get("p1_depol",0), ncfg.get("p2_depol",0),
                                  ncfg.get("p_amp",0), ncfg.get("readout_p01",0),
                                  ncfg.get("readout_p10",0)], dtype=np.float32)
            x = np.concatenate([feats, noise_vec, np.array([y_tilde, var, shots_noisy], dtype=np.float32)])
            samples.append(
                Sample(
                    x=x,
                    y_noisy=y_tilde,
                    y_ideal=y_star,
                    meta={
                        "noise": dict(ncfg),
                        "qc": qc,
                        "observable": obs,
                        "circuit_index": circ_idx,
                        "shots_noisy": shots_noisy,
                        "shots_ideal": shots_ideal,
                    },
                )
            )
    return samples

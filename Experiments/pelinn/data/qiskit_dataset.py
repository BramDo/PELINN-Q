# pelinn/data/qiskit_dataset.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Dict, Tuple
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
# BackendEstimator existed in older/newer qiskit primitives with slightly
# different names across releases. Import robustly and provide a clear
# fallback/helpful error if neither is available.
try:
    from qiskit.primitives import BackendEstimator
except Exception:
    try:
        # Newer/alternate installations may expose a generic Estimator primitive
        from qiskit.primitives import Estimator as BackendEstimator
    except Exception:
        BackendEstimator = None
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

def _expectation_from_counts(obs: SparsePauliOp, counts: dict, shots: int) -> float:
    """Compute expectation value of a SparsePauliOp from measurement counts.

    This fallback supports observables composed of Pauli strings containing
    only 'I' and 'Z' terms (the common Z-measurement cases used in this repo).
    The bitstring keys returned by Aer are ordered with qubit-0 as the
    right-most character, so we map accordingly.
    """
    # Convert to list of (pauli_str, coeff) pairs
    try:
        terms = obs.to_list()
    except Exception:
        # Older qiskit versions expose paulis/coeffs differently
        terms = [(str(p), complex(c).real) for p, c in zip(obs.paulis, obs.coeffs)]

    total = 0.0
    for pauli_str, coeff in terms:
        pauli_str = pauli_str.strip()
        # Only support I/Z
        if any(ch not in ("I", "Z") for ch in pauli_str):
            raise NotImplementedError("Fallback only supports Pauli strings with I and Z.")
        # contribution from this term
        term_exp = 0.0
        for bitstr, cnt in counts.items():
            # bitstr is like '0101' with qubit-0 at right-most char
            sign = 1.0
            # for each Z in pauli_str, map its index to bit position
            for idx, ch in enumerate(pauli_str):
                if ch == "Z":
                    # pauli_str index 0 corresponds to qubit 0
                    qubit_idx = idx
                    bitpos = len(bitstr) - 1 - qubit_idx
                    b = int(bitstr[bitpos])
                    sign *= -1.0 if b == 1 else 1.0
            term_exp += sign * cnt
        term_exp = term_exp / max(1, shots)
        total += float(complex(coeff).real) * term_exp
    return float(total)


def estimate_expectation(qc: QuantumCircuit, obs: SparsePauliOp, estimator: object | None, shots: int | None, backend: AerSimulator | None = None) -> Tuple[float, float]:
    """Estimate expectation value and variance using either a primitives estimator
    (when available) or a fallback that uses statevector (ideal) or counts.

    Parameters
    - qc: QuantumCircuit to evaluate
    - obs: SparsePauliOp observable
    - estimator: a BackendEstimator/Estimator instance or None
    - shots: number of shots (None or 0 => ideal)
    - backend: AerSimulator instance to use for counts fallback (optional)
    """
    # If a primitives estimator is provided, use it (handle its result shapes)
    if estimator is not None:
        runner = estimator.run(circuits=qc, observables=obs, parameter_values=None, shots=shots)
        try:
            res = runner.result()
        except Exception:
            res = runner
        # extract expectation
        ev = None
        if hasattr(res, "values"):
            try:
                ev = float(res.values[0])
            except Exception:
                ev = None
        if ev is None:
            try:
                first = res[0]
                try:
                    ev = float(first)
                except Exception:
                    ev = float(first.values[0])
            except Exception:
                ev = None
        if ev is None:
            raise ValueError("Unable to extract expectation from estimator result.")
        # variance from metadata if present
        var = 0.0
        try:
            if hasattr(res, "metadata") and len(res.metadata) > 0:
                meta0 = res.metadata[0]
                if isinstance(meta0, dict):
                    var = float(meta0.get("variance", 0.0))
                else:
                    var = float(getattr(meta0, "variance", 0.0))
        except Exception:
            var = 0.0
        return ev, var

    # Fallback path (no primitives estimator available)
    # Ideal expectation (statevector)
    if shots in (0, None):
        try:
            from qiskit.quantum_info import Statevector

            sv = Statevector.from_instruction(qc)
            ev = float(sv.expectation_value(obs).real)
            return ev, 0.0
        except Exception:
            # Fall through to shot-based sampling if statevector fails
            pass

    # Shot-based counts fallback: run AerSimulator if provided or create one
    if backend is None:
        backend = AerSimulator()
    # If we intend to run shots but the circuit has no measurements, add them
    qc_for_run = qc
    if shots and shots > 0:
        try:
            has_measure = "measure" in qc.count_ops()
        except Exception:
            has_measure = False
        if not has_measure:
            qc_for_run = qc.copy()
            qc_for_run.measure_all()

    compiled = transpile(qc_for_run, backend)
    job = backend.run(compiled, shots=int(shots) if shots and shots > 0 else 1)
    res = job.result()
    counts = {}
    # try several ways to extract counts from the result object
    try:
        counts = res.get_counts()
    except Exception:
        try:
            # older result format: inspect first result entry
            r0 = res.results[0]
            data = getattr(r0, "data", None) or r0.get("data", {})
            if isinstance(data, dict):
                counts = data.get("counts") or data.get("measurement_counts") or {}
        except Exception:
            counts = {}
    shots_used = shots if shots and shots > 0 else sum(counts.values()) if counts else 1
    ev = _expectation_from_counts(obs, counts, shots_used)
    # Compute simple variance estimate for Bernoulli-like outcome (rough)
    var = 0.0
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

    est_ideal = BackendEstimator(backend_ideal) if BackendEstimator is not None else None
    samples = []
    for circ_idx, (qc, obs) in enumerate(zip(circuits, observables)):
        feats = circuit_features(qc)
        # Ideal label
        y_star, _ = estimate_expectation(qc, obs, est_ideal, shots=None if shots_ideal in (0, None) else shots_ideal, backend=backend_ideal)

        for ncfg in noise_grid:
            key = tuple(sorted(ncfg.items()))
            if key not in backend_noisy_cache:
                nm = make_noise_model(**ncfg)
                backend = AerSimulator(noise_model=nm)
                backend_noisy_cache[key] = (backend, BackendEstimator(backend) if BackendEstimator is not None else None)
            b, est_noisy = backend_noisy_cache[key]

            qc_noisy = transpile(qc, b)
            y_tilde, var = estimate_expectation(qc_noisy, obs, est_noisy, shots=shots_noisy, backend=b)

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

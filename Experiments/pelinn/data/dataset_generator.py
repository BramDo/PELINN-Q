import numpy as np
import torch as th
import pandas as pd

import ast
import random
import json
from typing import Union, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from tqdm import tqdm

from qiskit import QuantumCircuit, transpile
from qiskit import qasm3

from qiskit.circuit.random import random_clifford_circuit, random_circuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import real_amplitudes, efficient_su2, n_local, qaoa_ansatz
try:
    from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
except Exception:  # pragma: no cover - older qiskit versions
    get_standard_gate_name_mapping = None

from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info import Operator

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit_aer.primitives import SamplerV2 as Sampler
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    amplitude_damping_error,
    phase_damping_error,
    thermal_relaxation_error,
    ReadoutError,
)
from qiskit_aer.noise.noiseerror import NoiseError
from qiskit_aer.noise.errors import pauli_error, kraus_error
from qiskit_aer.noise.errors.standard_errors import coherent_unitary_error

from qiskit_ibm_runtime import QiskitRuntimeService


# ------------------
# Circuit generation
# ------------------

class QuantumCircuitGenerator:
    """
    A unified class for generating various types of quantum circuits.
    
    This class provides methods to generate:
    - Random quantum circuits
    - Random Clifford circuits with fixed depth
    - QAOA circuits with random or custom cost operators
    - Parametric circuits (both custom and built-in ansätze)
    """
    
    def __init__(self, n_qubits: int, depth: int, seed: Optional[int] = None):
        """
        Initialize the QuantumCircuitGenerator.
        
        Args:
            n_qubits (int): Number of qubits in the circuits
            depth (int): Depth of the circuits
            seed (int, optional): Global seed for reproducibility. Defaults to None.
        """
        self.n_qubits = n_qubits
        self.depth = depth
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def generate(
        self, 
        circuit_type: str, 
        n_qubits: Optional[int] = None,
        depth: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> QuantumCircuit:
        """
        Generate a quantum circuit based on the specified type.
        
        Args:
            circuit_type (str): Type of circuit to generate. Options:
                - "random": Random quantum circuit
                - "random_clifford": Random Clifford circuit with fixed depth
                - "qaoa": QAOA circuit
                - "variational": Parametric/variational circuit
            n_qubits (int, optional): Number of qubits. Uses instance value if None.
            depth (int, optional): Circuit depth. Uses instance value if None.
            seed (int, optional): Random seed. Uses instance value if None.
            **kwargs: Additional arguments specific to each circuit type:
                
                For "qaoa":
                    - n_terms (int): Number of operators in the cost operator
                    - seed_cost_op (int, optional): Seed for cost operator generation
                    - seed_parameters (int, optional): Seed for parameter generation
                    - random_cost_operator (bool, optional): If True, generate random cost operator
                    - cost_operator (SparsePauliOp, optional): Custom cost operator
                
                For "variational":
                    - ansatz_type (str, optional): "custom" or "implemented"
                    - ready_ansatz_name (str, optional): Name of built-in ansatz
                    - rotation_blocks (str, optional): Rotation gates for custom circuit
                    - entanglement (str | list, optional): Entanglement pattern
        
        Returns:
            QuantumCircuit: The generated quantum circuit
        
        Raises:
            ValueError: If invalid circuit_type is provided
        """
        # Use provided values or fall back to instance defaults
        _n_qubits = n_qubits if n_qubits is not None else self.n_qubits
        _depth = depth if depth is not None else self.depth
        _seed = seed if seed is not None else self.seed
        
        if circuit_type == "random":
            return self.generate_random_circuit(n_qubits=_n_qubits, depth=_depth, seed=_seed)
        
        elif circuit_type == "random_clifford":
            return self.generate_clifford_circuit(n_qubits=_n_qubits, depth=_depth, seed=_seed)
        
        elif circuit_type == "qaoa":
            # Extract QAOA-specific parameters
            n_terms = kwargs.get('n_terms')
            if n_terms is None:
                raise ValueError("QAOA circuit requires 'n_terms' parameter")
            
            seed_cost_op = kwargs.get('seed_cost_op', None)
            seed_parameters = kwargs.get('seed_parameters', None)
            random_cost_operator = kwargs.get('random_cost_operator', True)
            cost_operator = kwargs.get('cost_operator', None)
            
            return self.generate_qaoa_circuit(
                n_qubits=_n_qubits,
                depth=_depth,
                n_terms=n_terms,
                seed_cost_op=seed_cost_op,
                seed_parameters=seed_parameters,
                random_cost_operator=random_cost_operator,
                cost_operator=cost_operator
            )
        
        elif circuit_type == "variational":
            # Extract variational circuit parameters
            ansatz_type = kwargs.get('ansatz_type', 'custom')
            ready_ansatz_name = kwargs.get('ready_ansatz_name', 'real_amplitudes')
            rotation_blocks = kwargs.get('rotation_blocks', 'ry')
            entanglement = kwargs.get('entanglement', 'linear')
            
            return self.generate_parametric_circuit(
                n_qubits=_n_qubits,
                depth=_depth,
                ansatz_type=ansatz_type,
                ready_ansatz_name=ready_ansatz_name,
                rotation_blocks=rotation_blocks,
                entanglement=entanglement,
                seed=_seed
            )
        
        else:
            raise ValueError(
                f"Unknown circuit type: {circuit_type}. "
                f"Valid options are: 'random', 'random_clifford', 'qaoa', 'variational'"
            )
 
    def generate_random_circuit(self, n_qubits: Optional[int] = None, depth: Optional[int] = None, seed: Optional[int] = None) -> QuantumCircuit:
        """
        Generate a random quantum circuit.
        
        Args:
            n_qubits (int, optional): Number of qubits. Uses instance value if None.
            depth (int, optional): Circuit depth. Uses instance value if None.
            seed (int, optional): Seed for reproducibility. Uses class seed if None.
        
        Returns:
            QuantumCircuit: Output circuit
        """
        _n_qubits = n_qubits if n_qubits is not None else self.n_qubits
        _depth = depth if depth is not None else self.depth
        _seed = seed if seed is not None else self.seed
        qc = random_circuit(_n_qubits, _depth, seed=_seed)
        return qc
    
    def generate_clifford_circuit(self, n_qubits: Optional[int] = None, depth: Optional[int] = None, seed: Optional[int] = None) -> QuantumCircuit:
        """
        Generate a random Clifford circuit with exactly fixed depth.
        
        Args:
            n_qubits (int, optional): Number of qubits. Uses instance value if None.
            depth (int, optional): Circuit depth. Uses instance value if None.
            seed (int, optional): Seed for reproducibility. Uses class seed if None.
        
        Returns:
            QuantumCircuit: Output circuit
        """
        _n_qubits = n_qubits if n_qubits is not None else self.n_qubits
        _depth = depth if depth is not None else self.depth
        _seed = seed if seed is not None else self.seed
        
        if _seed is not None:
            np.random.seed(_seed)
        
        qc = QuantumCircuit(_n_qubits)
        layers = 0
        gate_counter = 0
        
        while layers < _depth:
            occupied = set()
            
            while True:
                # Sample exactly one random Clifford gate
                cliff = random_clifford_circuit(
                    num_qubits=_n_qubits,
                    num_gates=1,
                    seed=None if _seed is None else _seed + gate_counter
                )
                op, qargs = cliff.data[0].operation, cliff.data[0].qubits
                
                # If any of the qubits for this gate is already used in layer → stop layer
                if any(q in occupied for q in qargs):
                    break
                
                # Append gate using qubit objects directly
                qc.append(op, qargs)
                
                # Mark those qubits as used in this layer
                occupied.update(qargs)
                gate_counter += 1
            
            layers += 1
        
        return qc
    
    def generate_qaoa_circuit(
        self,
        n_qubits: Optional[int] = None,
        depth: Optional[int] = None,
        n_terms: int = None,
        seed_cost_op: Optional[int] = None,
        seed_parameters: Optional[int] = None,
        random_cost_operator: bool = True,
        cost_operator: Optional[SparsePauliOp] = None
    ) -> QuantumCircuit:
        """
        Generate a random circuit with the architecture of a QAOA circuit.
        
        Args:
            n_qubits (int, optional): Number of qubits. Uses instance value if None.
            depth (int, optional): Circuit depth. Uses instance value if None.
            n_terms (int): Number of operators in the cost operator
            seed_cost_op (int, optional): Seed for random generation of cost operator
            seed_parameters (int, optional): Seed for random generation of parameters
            random_cost_operator (bool, optional): If True, generate cost operator randomly
            cost_operator (SparsePauliOp, optional): Custom cost operator for QAOA
        
        Returns:
            QuantumCircuit: The quantum circuit
        
        Raises:
            ValueError: If no cost operator is provided
        """
        _n_qubits = n_qubits if n_qubits is not None else self.n_qubits
        _depth = depth if depth is not None else self.depth
        
        if random_cost_operator:
            cost_op = self._random_diagonal_cost_operator(
                n_qubits=_n_qubits,
                n_terms=n_terms,
                seed=seed_cost_op if seed_cost_op is not None else self.seed
            )
        else:
            cost_op = cost_operator
        
        if cost_op is None:
            raise ValueError("No provided cost operator. Please give one in input or use the random generator")
        
        circuit = qaoa_ansatz(cost_op, reps=_depth)
        
        param_seed = seed_parameters if seed_parameters is not None else self.seed
        rng = np.random.default_rng(param_seed)
        random_values = rng.uniform(0, 1, size=len(circuit.parameters))
        circuit = circuit.assign_parameters(random_values)
        
        return circuit
    
    def generate_parametric_circuit(
        self,
        n_qubits: Optional[int] = None,
        depth: Optional[int] = None,
        ansatz_type: str = "custom",
        ready_ansatz_name: str = "real_amplitudes",
        rotation_blocks: str = "ry",
        entanglement: Union[str, List[Tuple[int, int]]] = "linear",
        seed: Optional[int] = None
    ) -> QuantumCircuit:
        """
        Generate a parametric circuit with built-in method from Qiskit or custom architecture.
        
        Args:
            n_qubits (int, optional): Number of qubits. Uses instance value if None.
            depth (int, optional): Circuit depth. Uses instance value if None.
            ansatz_type (str, optional): "custom" or "implemented". Defaults to "custom".
            ready_ansatz_name (str, optional): Name of built-in ansatz. Defaults to "real_amplitudes".
            rotation_blocks (str, optional): Rotation gates for custom circuit. Defaults to "ry".
            entanglement (str | list, optional): Entanglement pattern. Defaults to "linear".
            seed (int, optional): Seed for random generator. Uses class seed if None.
        
        Returns:
            QuantumCircuit: The parametric quantum circuit
        
        Raises:
            ValueError: If invalid ansatz_type or ready_ansatz_name
        """
        _n_qubits = n_qubits if n_qubits is not None else self.n_qubits
        _depth = depth if depth is not None else self.depth
        _seed = seed if seed is not None else self.seed
        
        # 1. READY-MADE ANSATZ
        if ansatz_type == "implemented":
            if ready_ansatz_name == "real_amplitudes":
                circ = real_amplitudes(
                    num_qubits=_n_qubits,
                    reps=_depth,
                    entanglement=entanglement
                )
            elif ready_ansatz_name == "efficent_su2":
                circ = efficient_su2(
                    num_qubits=_n_qubits,
                    reps=_depth,
                    entanglement=entanglement
                )
            elif ready_ansatz_name == "n_local":
                circ = n_local(
                    num_qubits=_n_qubits,
                    rotation_blocks=rotation_blocks,
                    entanglement_blocks="cx",
                    entanglement=_depth,
                    reps=_depth
                )
            else:
                raise ValueError(f"Unknown implemented ansatz {ready_ansatz_name}")
            
            params = list(circ.parameters)
        
        # 2. CUSTOM ANSATZ
        elif ansatz_type == "custom":
            circ = QuantumCircuit(_n_qubits)
            params = []
            
            for layer in range(_depth):
                self._add_rotations(layer, circ, _n_qubits, rotation_blocks, params)
                self._add_entanglement(circ, _n_qubits, entanglement)
        else:
            raise ValueError(f"Incorrect ansatz type: {ansatz_type}, use custom or implemented")
        
        # 3. RANDOM PARAMETERS
        if _seed is not None:
            np.random.seed(_seed)
        
        init_values = {p: float(np.random.rand()) for p in params}
        circuit = circ.assign_parameters(init_values)
        
        return circuit
    
    #Private helpers method
    
    @staticmethod
    def _random_diagonal_cost_operator(
        n_qubits: int,
        n_terms: int,
        seed: Optional[int] = None,
        coeff: float = 1.0
    ) -> SparsePauliOp:
        """
        Generate a random diagonal SparsePauliOp (only I and Z) with fixed coefficients.
        
        Args:
            n_qubits (int): Number of qubits
            n_terms (int): Number of Pauli strings in the cost operator
            seed (int, optional): Random seed for reproducibility
            coeff (float, optional): Coefficient for all terms. Defaults to 1.0.
        
        Returns:
            SparsePauliOp: Random diagonal cost operator
        """
        rng = np.random.default_rng(seed)
        labels = ['I', 'Z']
        terms = []
        
        while len(terms) < n_terms:
            # Generate a random string of I and Z
            pauli_string = ''.join(rng.choice(labels, size=n_qubits))
            # Skip full identity
            if 'Z' in pauli_string:
                terms.append(pauli_string)
        
        coeffs = [coeff] * n_terms
        return SparsePauliOp(terms, coeffs)
    
    @staticmethod
    def _add_rotations(layer: int, circ: QuantumCircuit, n_qubits: int, rotation_block: str, params: list):
        """Add a rotation layer with the specified block type."""
        if rotation_block == "ry":
            for q in range(n_qubits):
                p = Parameter(f"θ_{layer}_{q}")
                circ.ry(p, q)
                params.append(p)
        
        elif rotation_block == "rx_ry":
            for q in range(n_qubits):
                px = Parameter(f"θx_{layer}_{q}")
                py = Parameter(f"θy_{layer}_{q}")
                circ.rx(px, q)
                circ.ry(py, q)
                params.extend([px, py])
        
        elif rotation_block == "full_u3":
            for q in range(n_qubits):
                pa = Parameter(f"θa_{layer}_{q}")
                pb = Parameter(f"θb_{layer}_{q}")
                pc = Parameter(f"θc_{layer}_{q}")
                circ.u(pa, pb, pc, q)
                params.extend([pa, pb, pc])
        else:
            raise ValueError(f"Unknown rotation block: {rotation_block}")
    
    @staticmethod
    def _add_entanglement(circ: QuantumCircuit, n_qubits: int, entanglement: Union[str, List[Tuple[int, int]]]):
        """Add entangling CX gates according to the specified pattern."""
        if entanglement == "linear":
            for i in range(n_qubits - 1):
                circ.cx(i, i + 1)
        
        elif entanglement == "full":
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    circ.cx(i, j)
        
        elif entanglement == "circular":
            for i in range(n_qubits):
                circ.cx(i, (i + 1) % n_qubits)
        
        elif isinstance(entanglement, list):
            # Custom list of pairs like [(0,2), (1,3)]
            for (i, j) in entanglement:
                circ.cx(i, j)
        else:
            raise ValueError(f"Unknown entanglement pattern: {entanglement}")


# ------------------
# Circuit execution
# ------------------

@dataclass
class SamplerOutput:
    bitstrings: list
    shots_probs: list
    probs: np.array
   
   
@dataclass
class EstimatorOutput:
    observables: SparsePauliOp
    value: np.ndarray
    shots: int


def execute_circuit(
    circuit: QuantumCircuit,
    mode: str = "sampler",
    observables: List[SparsePauliOp] | None = None,
    shots: int = 1024,
    noiseless: bool = True,
    noise_model: NoiseModel | None = None,
    seed_simulation : int | None = None,
    opt_level_transpilation: int = 1
    ) -> Union[SamplerOutput, EstimatorOutput]:
    """Run the circuit through the Qiskit Aer Sampler or Estimator with or without a noise model.

    Args:
        circuit (QuantumCircuit): The quantum circuit that you want simulate
        mode (str, optional): Decide between the sampler or the Estimator. Defaults to "sampler".
        observables (List[SparsePauliOp] | None, optional): List of observables as Pauli strings. Defaults to None.
        shots (int, optional): Number of shots for the circuit simulation. Defaults to 1024.
        noiseless (bool, optional): Whether to run the circuit with or without noise. Defaults to True.
        noise_model (NoiseModel | None, optional): Noise model for the noisy simualation. Defaults to None.
        seed_simulation (int | None, optional): Seed for reproducibility. Defaults to None.
        opt_level_transpilation(int, optional): Optimization level for the tranpsilation of the circuit. Defaults to 1.

    Raises:
        ValueError: If noisless = True you must provide a noise model
        ValueError: If mode = Estimator you must provide observables
        ValueError: The dimension of the observables must be equal to the dimension of the circuit
        ValueError: If you provide neither a Sampler nor Estimator as mode input

    Returns:
        Union[SamplerOutput, EstimatorOutput]: Results of the simulation. It depends on the chosen mode
    """

    # ------------------- Backend -------------------
    if noiseless:
        backend = AerSimulator()
    else: 
        if noise_model is None:
            raise ValueError("No noise model is provided. Please give in input a noise model to execute the noisy circuit")
        backend = AerSimulator(noise_model=noise_model)

    # ------------------- SAMPLER -------------------
    if mode == "sampler":
        qc_meas = circuit.copy()
        if not qc_meas.cregs:
            qc_meas.measure_all()

        # transpile for optimization
        qc_meas = transpile(qc_meas, backend=backend, optimization_level=opt_level_transpilation)

        sampler = Sampler().from_backend(backend=backend, seed=seed_simulation)
        
        pub = (qc_meas)
        results = sampler.run([pub], shots=shots).result()[0].data.meas.get_counts()

        bitstrings = list(results.keys())
        shots_probs = list(results.values())
        probs = np.array(shots_probs) / shots

        return SamplerOutput(bitstrings=bitstrings,
                             shots_probs=shots_probs,
                             probs=probs)

    # ------------------- ESTIMATOR -------------------
    if mode == "estimator":
        if observables is None:
            raise ValueError("Estimator mode requires an observable.")
        for obs in observables:
            if obs.num_qubits != circuit.num_qubits:
                raise ValueError("Observable and circuit must have same number of qubits.")

        # transpile for optimization
        qc_transpiled = transpile(circuit, backend=backend, optimization_level=opt_level_transpilation)

        estimator = Estimator().from_backend(backend=backend) 
    
        pub = (qc_transpiled, observables)
        result = estimator.run([pub]).result()[0].data.evs

        return EstimatorOutput(
            observables=observables,
            value=result,
            shots=shots
        )
    else:
        raise ValueError(f"Unknown simulation mode {mode}. Choose between sampler or estimator")
    
 # ------------------


# ------------------
# Noise model
# ------------------

# Default noise profile used when noise_list is empty.
DEFAULT_NOISE_LEVELS = {
    "p1_depol": 0.01,
    "p2_depol": 0.05,
    "readout": 0.05,
}


def _default_noise_list(num_qubits: int) -> List[Dict[str, Any]]:
    noise_list: List[Dict[str, Any]] = []
    for q in range(num_qubits):
        noise_list.append({"type": "depolarizing", "p": DEFAULT_NOISE_LEVELS["p1_depol"], "qubits": [q]})
    for q in range(max(0, num_qubits - 1)):
        noise_list.append(
            {"type": "correlated_depolarizing", "p": DEFAULT_NOISE_LEVELS["p2_depol"], "qubits": [q, q + 1], "gates": ["cx"]}
        )
    noise_list.append(
        {
            "type": "readout",
            "p0to1": DEFAULT_NOISE_LEVELS["readout"],
            "p1to0": DEFAULT_NOISE_LEVELS["readout"],
            "qubits": list(range(num_qubits)),
        }
    )
    return noise_list


def _apply_default_noise_config(noise_config: Dict[str, Any], num_qubits: int) -> Dict[str, Any]:
    if not isinstance(noise_config, dict):
        return noise_config
    if noise_config.get("disable_default_noise"):
        return noise_config
    noise_list = noise_config.get("noise_list")
    if not noise_list:
        merged = dict(noise_config)
        merged["noise_list"] = _default_noise_list(num_qubits)
        return merged
    return noise_config


class NoiseModelFactory:
    def __init__(self, num_qubits: int, simulator : AerSimulator | None = None, seed_noise: int | None = None):
        self.num_qubits = num_qubits
        self.seed_noise = seed_noise

        # set seeds for deterministic construction where relevant
        if seed_noise is not None:
            np.random.seed(seed_noise)
            random.seed(seed_noise)

        # Aer simulator to query basis_gates
        if simulator is not None:
            self.simulator = simulator
        else:
            self.simulator = AerSimulator(seed_simulator=seed_noise) if seed_noise is not None else AerSimulator()
        self.basis_gates = set(self.simulator.configuration().basis_gates)
        self._gate_qubits: Dict[str, int] = {}
        if get_standard_gate_name_mapping is not None:
            try:
                mapping = get_standard_gate_name_mapping()
                self._gate_qubits = {name: inst.num_qubits for name, inst in mapping.items()}
            except Exception:
                self._gate_qubits = {}

    # Helpers
    def _ensure_gate_supported(self, gate: str):
        if gate not in self.basis_gates:
            raise ValueError(f"Gate '{gate}' is not in AerSimulator basis_gates: {sorted(self.basis_gates)}")

    def _all_qubits_list(self):
        return list(range(self.num_qubits))

    def _gate_num_qubits(self, gate: str) -> int | None:
        return self._gate_qubits.get(gate)

    # Main builder from config
    def build_from_config(self, config: Dict[str, Any]) -> NoiseModel:
        """
        Build a NoiseModel from a configuration dictionary.

        config example:
        {
          "noise_list": [
            {"type": "depolarizing", "p": 0.01, "qubits": [0], "gates": ["x"]},
            {"type": "thermal_relaxation", "t1": 50e3, "t2": 70e3, "gate_time": 50, "qubits": [1], "gates": ["x"]},
            {"type": "correlated_depolarizing", "p": 0.03, "qubits": [0,1], "gates": ["cx"]},
            {"type": "crosstalk", "control_gate": "cx", "victim_qubits": [3], "p": 0.02},
            {"type": "readout", "p0to1": 0.02, "p1to0": 0.03, "qubits": [0,1,2]}
          ]
        }
        """
        nm = NoiseModel()

        # keep track to set seed for reproduction (Aer accepts seeds at simulator level;
        # setting _default_options is a pragmatic choice to pass seed_noise through the model)
        if self.seed_noise is not None:
            try:
                nm._default_options = {"seed_noise": int(self.seed_noise)}
            except Exception:
                # non-fatal if internals differ across qiskit-aer versions
                pass

        noise_list = config.get("noise_list", [])

       

        for entry in noise_list:

            """
            # default qubits & gates
            qubits: List[int] = entry.get("qubits", self._all_qubits_list())
            gates: List[str] = entry.get("gates", None)  # None => use gate-specific defaults

            # ---------- SANITY CHECK ----------
            invalid_qubits = [q for q in qubits if q < 0 or q >= self.num_qubits]
            if invalid_qubits:
                raise ValueError(f"Invalid qubits {invalid_qubits} in noise entry {entry}. "
                                    f"Valid qubits are 0..{self.num_qubits-1}")
            """
        
            typ = entry.get("type")
            # default qubits & gates
            qubits: List[int] = entry.get("qubits", self._all_qubits_list())
            gates: List[str] = entry.get("gates", None)  # None => use gate-specific defaults

            # ---------- Depolarizing ----------
            if typ == "depolarizing":
                # Example input: {"type": "depolarizing", "p": 0.01, "qubits": [0], "gates": ["x"]}
                p = float(entry["p"])
                err_qubits = len(qubits) if len(qubits) > 1 else 1
                err = depolarizing_error(p, err_qubits)
                target_gates = list(self.basis_gates) if gates is None else list(gates)
                for g in target_gates:
                    if gates is not None:
                        self._ensure_gate_supported(g)
                    gate_qubits = self._gate_num_qubits(g)
                    if gate_qubits is not None and gate_qubits != err_qubits:
                        continue
                    try:
                        nm.add_quantum_error(err, g, qubits if len(qubits) > 1 else [qubits[0]])
                    except NoiseError:
                        continue

            # ---------- Amplitude damping ----------
            elif typ == "amplitude_damping":
                # Example input: {"type": "amplitude_damping", "gamma": 0.02, "qubits": [1], "gates": ["x"]}
                gamma = float(entry["gamma"])
                target_gates = gates or ["x", "sx"]
                for g in target_gates:
                    if g in self.basis_gates:
                        err = amplitude_damping_error(gamma, canonical_kraus=False)
                        for q in qubits:
                            nm.add_quantum_error(err, g, [q])

            # ---------- Phase damping ----------
            elif typ == "phase_damping":
                # Example input: {"type": "phase_damping", "lambda": 0.05, "qubits": [0], "gates": ["rz"]}
                lam = float(entry["lambda"])
                target_gates = gates or ["rz", "u1", "u"]  # example common phase-like gates
                for g in target_gates:
                    if g in self.basis_gates:
                        err = phase_damping_error(lam, canonical_kraus=False)
                        for q in qubits:
                            nm.add_quantum_error(err, g, [q])

            # ---------- Thermal relaxation ----------
            elif typ == "thermal_relaxation":
                # Example input: {"type": "thermal_relaxation", "t1": 50e3, "t2": 70e3, "gate_time": 50, "qubits": [1], "gates": ["x"]}
                t1 = float(entry["t1"])
                t2 = float(entry["t2"])
                gate_time = float(entry["gate_time"])
                target_gates = gates or ["x", "sx", "cx"]
                for g in target_gates:
                    if g in self.basis_gates:
                        # thermal_relaxation_error returns a 1-qubit error; for 2-qubit gate tensor two of them
                        if entry.get("multi_qubit", False) and len(qubits) > 1:
                            t1_err = thermal_relaxation_error(t1, t2, gate_time)
                            two_q = t1_err.tensor(thermal_relaxation_error(t1, t2, gate_time))
                            nm.add_quantum_error(two_q, g, qubits)
                        else:
                            err = thermal_relaxation_error(t1, t2, gate_time)
                            for q in qubits:
                                nm.add_quantum_error(err, g, [q])

            # ---------- Pauli error ----------
            elif typ == "pauli":
                # Example input: {"type": "pauli", "paulis": [("X",0.1), ("I",0.9)], "gates":["cx"], "qubits":[0,1]}
                paulis = entry["paulis"]  # list of tuples or strings with probabilities
                # pauli_error expects list of (label, prob) e.g. [("II", 0.9), ("XI", 0.1)]
                err = pauli_error(paulis)
                target_gates = gates or (["cx"] if len(qubits) > 1 else ["x"])
                for g in target_gates:
                    if g in self.basis_gates:
                        nm.add_quantum_error(err, g, qubits if len(qubits) > 1 else [qubits[0]])

            # ---------- Kraus error ----------
            elif typ == "kraus":
                # entry must include "kraus_ops": a python-list-of-ndarrays (or strings that literal_eval -> lists)
                # Example input: {"type": "kraus", "kraus_ops": [[[1,0],[0,0]], [[0,0],[0,1]]], "gates":["x"], "qubits":[0]}
                kraus_ops = entry["kraus_ops"]
                # kraus_error expects list of 2D numpy arrays
                kraus_ops_np = [np.array(op) for op in kraus_ops]
                err = kraus_error(kraus_ops_np)
                target_gates = gates or ["x"]
                for g in target_gates:
                    if g in self.basis_gates:
                        for q in qubits:
                            nm.add_quantum_error(err, g, [q])

            # ---------- Coherent unitary ----------
            elif typ == "coherent_unitary":
                # entry: {"unitary": <ndarray or nested list>, "gates":["x"], "qubits":[0]}
                # Example input: {"type": "coherent_unitary", "unitary": [[1,0],[0,-1]], "gates":["x"], "qubits":[0]}
                U = np.array(entry["unitary"])
                op = Operator(U)
                err = coherent_unitary_error(op)
                target_gates = gates or ["x"]
                for g in target_gates:
                    if g in self.basis_gates:
                        if len(qubits) > 1:
                            nm.add_quantum_error(err, g, qubits)
                        else:
                            nm.add_quantum_error(err, g, [qubits[0]])

            # ---------- Correlated depolarizing ----------
            elif typ == "correlated_depolarizing":
                # Example input: {"type": "correlated_depolarizing", "p": 0.03, "qubits":[0,1], "gates":["cx"]}
                p = float(entry["p"])
                if len(qubits) < 2:
                    raise ValueError("correlated_depolarizing requires multiple qubits")
                err = depolarizing_error(p, len(qubits))
                target_gates = gates or ["cx"]
                for g in target_gates:
                    if g in self.basis_gates:
                        nm.add_quantum_error(err, g, qubits)

            # ---------- Correlated Pauli ----------
            elif typ == "correlated_pauli":
                # entry: { "paulis":[("XX", 0.1), ("II",0.9)], "gates":["cx"], "qubits":[0,1] }
                # Example input: {"type": "correlated_pauli", "paulis":[("XX",0.1),("II",0.9)], "gates":["cx"], "qubits":[0,1]}
                paulis = entry["paulis"]
                err = pauli_error(paulis)
                target_gates = gates or ["cx"]
                for g in target_gates:
                    if g in self.basis_gates:
                        nm.add_quantum_error(err, g, qubits)

            # ---------- Crosstalk ----------
            elif typ == "crosstalk":
                # entry: {"control_gate":"cx", "victim_qubits":[3], "p":0.02, "victim_gate":"x"}
                # Example input: {"type": "crosstalk", "control_gate":"cx", "victim_qubits":[3], "p":0.02, "victim_gate":"x"}
                control_gate = entry["control_gate"]
                victim_qubits = entry.get("victim_qubits", [])
                p = float(entry.get("p", 0.01))
                victim_gate = entry.get("victim_gate", None)
                # produce a single-qubit depolarizing error on victims triggered by control_gate
                xerr = depolarizing_error(p, 1)
                # validation
                if control_gate not in self.basis_gates:
                    raise ValueError(f"control_gate '{control_gate}' not supported by Aer basis_gates")
                # the triggered gate attached to victim can be any basis gate or left implicit as the trigger
                if victim_gate is None:
                    # attach triggered error to the control gate but specifying victim qubit index
                    for v in victim_qubits:
                        nm.add_quantum_error(xerr, [control_gate], [v])
                else:
                    self._ensure_gate_supported(victim_gate)
                    for v in victim_qubits:
                        nm.add_quantum_error(xerr, [victim_gate], [v])

            # ---------- Readout ----------
            elif typ == "readout":
                # Example input: {"type": "readout", "p0to1":0.02, "p1to0":0.03, "qubits":[0,1,2]}
                p0to1 = float(entry.get("p0to1", 0.02))
                p1to0 = float(entry.get("p1to0", 0.02))
                target_qubits = entry.get("qubits", self._all_qubits_list())
                mat = [[1 - p0to1, p0to1], [p1to0, 1 - p1to0]]
                rerr = ReadoutError(mat)
                for q in target_qubits:
                    nm.add_readout_error(rerr, [q])

            # ---------- From real backend ----------
            elif typ == "from_backend" or typ == "from_real_backend":
                # Example input: {"type": "from_backend", "backend_name":"ibmq_lima", "ibm_token":"..."}
                backend_name = entry["backend_name"]
                ibm_token = entry.get("ibm_token", None)
                # lazy import / runtime: use QiskitRuntimeService if token provided
                try: 
                    service = QiskitRuntimeService(token=ibm_token) if ibm_token else QiskitRuntimeService()
                    backend = service.backend(backend_name)
                except Exception as e:
                        raise RuntimeError("Could not load IBM backend: " + str(e))
                nm = NoiseModel.from_backend(backend)
                # reapply seed if present
                if self.seed_noise is not None:
                    try:
                        nm._default_options = {"seed_noise": int(self.seed_noise)}
                    except Exception:
                        pass

            else:
                raise ValueError(f"Unsupported noise type in config: '{typ}'")

        return nm


# Utility: load config from a simple TXT (key=value lines) or JSON
def load_config_from_txt_or_json(path: str) -> Dict[str, Any]:
    """
    If file endswith .json -> use json.load
    else parse simple key=value lines where values are python-literal (lists, dicts).
    """
    
    if path.lower().endswith(".json"):
        with open(path, "r") as fh:
            return json.load(fh)

    cfg = {}
    with open(path, "r") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            try:
                parsed = ast.literal_eval(val)
            except Exception:
                parsed = val
            cfg[key] = parsed
    return cfg


# ------------------
# Generate dataset
# ------------------

@dataclass
class DatasetSample:
    """Single sample in the quantum dataset for ML tasks."""
    sample_id: int
    x: List[float]  # Noisy observable values (input features)
    y: List[float]  # Noiseless observable values (target labels)
    circuit_qasm: Optional[str] = None
    circuit_depth: Optional[int] = None
    observables: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class QuantumDataset:
    """
    A class to store and manipulate quantum circuit datasets.
    
    Provides methods to convert to various formats (NumPy, PyTorch, Pandas)
    and save to disk in multiple formats.
    """
    
    def __init__(
        self,
        dataset: List[DatasetSample],
        n_qubits: int,
        n_observables: int,
        circuit_type: str,
        shots: int,
        seed: Optional[int],
        noise_config: Dict[str, Any],
        observables: List[SparsePauliOp]
    ):
        """
        Initialize the QuantumDataset.
        
        Args:
            dataset: List of DatasetSample objects
            n_qubits: Number of qubits
            n_observables: Number of observables
            circuit_type: Type of circuit
            shots: Number of shots used
            seed: Random seed used
            noise_config: Noise configuration
            observables: List of observables
        """
        self.dataset = dataset
        self.n_qubits = n_qubits
        self.n_observables = n_observables
        self.circuit_type = circuit_type
        self.shots = shots
        self.seed = seed
        self.noise_config = noise_config
        self.observables = observables
    
    def dataset_to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert dataset to numpy arrays.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (X, Y) where X is noisy, Y is noiseless
                X.shape = (n_samples, n_observables)
                Y.shape = (n_samples, n_observables)
        """
        if not self.dataset:
            raise ValueError("Dataset is empty.")
        
        X = np.array([sample.x for sample in self.dataset])
        Y = np.array([sample.y for sample in self.dataset])
        
        return X, Y
    
    def dataset_to_pandas(self, include_metadata: bool = False) -> pd.DataFrame:
        """
        Convert dataset to pandas DataFrame.
        
        Args:
            include_metadata (bool): Whether to include metadata columns
            
        Returns:
            pd.DataFrame: Dataset as DataFrame with columns for each observable
        """
        if not self.dataset:
            raise ValueError("Dataset is empty.")
        
        data = []
        for sample in self.dataset:
            row = {"sample_id": sample.sample_id}
            
            # Add noisy observables (x)
            for i, val in enumerate(sample.x):
                row[f"x_obs_{i}"] = val
            
            # Add noiseless observables (y)
            for i, val in enumerate(sample.y):
                row[f"y_obs_{i}"] = val
            
            # Optionally add metadata
            if include_metadata and sample.metadata:
                row["circuit_depth"] = sample.circuit_depth
                row["circuit_type"] = sample.metadata.get("circuit_type")
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def dataset_to_torch(self, device: str = "cpu") -> Tuple[th.Tensor, th.Tensor]:
        """
        Convert dataset to PyTorch tensors.
        
        Args:
            device (str): Device to place tensors on ('cpu', 'cuda', 'mps')
            
        Returns:
            Tuple[th.Tensor, th.Tensor]: (X, Y) as PyTorch tensors
                X.shape = (n_samples, n_observables) - noisy
                Y.shape = (n_samples, n_observables) - noiseless
        """
        X_np, Y_np = self.dataset_to_numpy()
        
        X_torch = th.tensor(X_np, dtype=th.float32, device=device)
        Y_torch = th.tensor(Y_np, dtype=th.float32, device=device)
        
        return X_torch, Y_torch
    
    def save_dataset(self, save_path: str, format: str = "csv"):
        """
        Save the generated dataset to disk in various formats.
        
        Args:
            save_path (str): Path where to save the dataset
            format (str): Save format - options:
                - "csv": Comma-separated values
                - "pandas": Pandas pickle format
                - "torch": PyTorch tensor format (.pt)
                - "numpy": NumPy arrays (.npz)
                - "json": Complete JSON with all metadata
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "csv":
            df = self.dataset_to_pandas(include_metadata=True)
            df.to_csv(save_path, index=False)
            print(f"Dataset saved to {save_path} (CSV format)")
        
        elif format == "pandas":
            df = self.dataset_to_pandas(include_metadata=True)
            df.to_pickle(save_path)
            print(f"Dataset saved to {save_path} (Pandas pickle format)")
        
        elif format == "torch":
            X, Y = self.dataset_to_torch()
            th.save({"X": X, "Y": Y, "metadata": self._get_global_metadata()}, save_path)
            print(f"Dataset saved to {save_path} (PyTorch format)")
        
        elif format == "numpy":
            X, Y = self.dataset_to_numpy()
            np.savez(save_path, X=X, Y=Y, metadata=self._get_global_metadata())
            print(f"Dataset saved to {save_path} (NumPy format)")
        
        elif format == "json":
            dataset_dict = [asdict(sample) for sample in self.dataset]
            
            with open(save_path, 'w') as f:
                json.dump({
                    "metadata": self._get_global_metadata(),
                    "samples": dataset_dict
                }, f, indent=2)
            
            print(f"Dataset saved to {save_path} (JSON format)")
        
        else:
            raise ValueError(f"Unsupported format: {format}. "
                           f"Use 'csv', 'pandas', 'torch', 'numpy', or 'json'.")
     
    def _get_global_metadata(self) -> Dict[str, Any]:
        """Get global metadata about the dataset."""
        return {
            "n_samples": len(self.dataset),
            "n_qubits": self.n_qubits,
            "n_observables": self.n_observables,
            "circuit_type": self.circuit_type,
            "shots": self.shots,
            "seed": self.seed,
            "noise_config": self.noise_config,
            "observables": [str(obs) for obs in self.observables]
        }


class GenerateQuantumDataset:
    """
    A class to generate quantum circuit datasets for machine learning.
    
    Each sample contains:
    - x: noisy observable expectation values (input)
    - y: noiseless observable expectation values (target)
    
    The class integrates:
    - QuantumCircuitGenerator for creating various circuit types
    - NoiseModelFactory for applying realistic noise
    - execute_circuit for running simulations
    """
    
    def __init__(
        self,
        n_qubits: int,
        depth: int,
        circuit_type: str,
        noise_config: Union[Dict[str, Any], str],
        observables: List[SparsePauliOp],
        circuit_params: Optional[Dict[str, Any]] = None,
        shots: int = 1024,
        seed: Optional[int] = None,
        save_circuits: bool = False,
        opt_level: int = 1
    ):
        """
        Initialize the dataset generator.
        
        Args:
            n_qubits (int): Number of qubits for all circuits
            depth (int): Depth of the circuit
            circuit_type (str): Type of circuit ("random", "random_clifford", "qaoa", "variational")
            noise_config (Dict or str): Noise configuration dict or path to config file
            observables (List[SparsePauliOp]): List of observables for measurements (REQUIRED)
            circuit_params (Dict, optional): Additional parameters for circuit generation
            shots (int, optional): Number of shots for simulation. Defaults to 1024.
            seed (int, optional): Global seed for reproducibility. Defaults to None.
            save_circuits (bool, optional): Whether to save circuit QASM. Defaults to False.
            opt_level (int, optional): Transpilation optimization level. Defaults to 1.
        
        Raises:
            ValueError: If observables list is empty or None
        """
        if observables is None or len(observables) == 0:
            raise ValueError("Observables are required for dataset generation. "
                           "Provide a list of SparsePauliOp objects.")
        
        self.n_qubits = n_qubits
        self.depth = depth
        self.circuit_type = circuit_type
        self.observables = observables
        self.n_observables = len(observables)
        self.circuit_params = circuit_params or {}
        self.shots = shots
        self.seed = seed
        self.save_circuits = save_circuits
        self.opt_level = opt_level
        
        # Load noise configuration
        if isinstance(noise_config, str):
            self.noise_config = load_config_from_txt_or_json(noise_config)
        else:
            self.noise_config = noise_config
        self.noise_config = _apply_default_noise_config(self.noise_config, n_qubits)
        
        # Initialize circuit generator
        self.circuit_generator = QuantumCircuitGenerator(
            n_qubits=n_qubits,
            depth=self.circuit_params.get('depth', self.depth),
            seed=seed
        )
        
        # Initialize noise model factory
        self.noise_factory = NoiseModelFactory(
            num_qubits=n_qubits,
            seed_noise=seed
        )
        
        # Build noise model from config
        self.noise_model = self.noise_factory.build_from_config(self.noise_config)
        
        # Set seeds for reproducibility
        if seed is not None:
            np.random.seed(seed)
            th.manual_seed(seed)
    
    def generate_single_sample(self, sample_id: int) -> DatasetSample:
        """
        Generate a single dataset sample with noisy (x) and noiseless (y) observable values.
        
        Args:
            sample_id (int): Unique identifier for this sample
            
        Returns:
            DatasetSample: Sample with x (noisy) and y (noiseless) observable values
        """
        # Generate circuit with unique seed per sample
        circuit_seed = None if self.seed is None else self.seed + sample_id
        
        circuit = self.circuit_generator.generate(
            circuit_type=self.circuit_type,
            seed=circuit_seed,
            **self.circuit_params
        )
        
        # Execute NOISELESS (target y)
        result_noiseless = execute_circuit(
            circuit=circuit,
            mode="estimator",
            observables=self.observables,
            shots=self.shots,
            noiseless=True,
            seed_simulation=circuit_seed,
            opt_level_transpilation=self.opt_level
        )
        
        # Execute NOISY (input x)
        result_noisy = execute_circuit(
            circuit=circuit,
            mode="estimator",
            observables=self.observables,
            shots=self.shots,
            noiseless=False,
            noise_model=self.noise_model,
            seed_simulation=circuit_seed,
            opt_level_transpilation=self.opt_level
        )
        
        # Extract observable values
        y = result_noiseless.value.tolist()  # Ground truth (noiseless)
        x = result_noisy.value.tolist()      # Noisy measurements (input)
        
        # Optional: save circuit info
        circuit_qasm = qasm3.dumps(circuit) if self.save_circuits else None
        circuit_depth = circuit.depth()
        obs_strings = [str(obs) for obs in self.observables]
        
        # Prepare metadata
        metadata = {
            "n_qubits": self.n_qubits,
            "circuit_type": self.circuit_type,
            "circuit_depth": circuit_depth,
            "shots": self.shots,
            "seed": circuit_seed
        }
        
        return DatasetSample(
            sample_id=sample_id,
            x=x,
            y=y,
            circuit_qasm=circuit_qasm,
            circuit_depth=circuit_depth,
            observables=obs_strings,
            metadata=metadata
        )
    
    def generate_dataset(
        self,
        n_samples: int,
        show_progress: bool = True
    ) -> QuantumDataset:
        """
        Generate a complete dataset with multiple samples.
        
        Args:
            n_samples (int): Number of samples to generate
            show_progress (bool, optional): Show progress bar. Defaults to True.
            
        Returns:
            QuantumDataset: Complete dataset object
        """
        dataset_samples = []
        
        iterator = tqdm(range(n_samples), desc="Generating dataset") if show_progress else range(n_samples)
        
        for i in iterator:
            try:
                sample = self.generate_single_sample(sample_id=i)
                dataset_samples.append(sample)
            except Exception as e:
                print(f"\nError generating sample {i}: {str(e)}")
                continue
        
        return QuantumDataset(
            dataset=dataset_samples,
            n_qubits=self.n_qubits,
            n_observables=self.n_observables,
            circuit_type=self.circuit_type,
            shots=self.shots,
            seed=self.seed,
            noise_config=self.noise_config,
            observables=self.observables
        ) 

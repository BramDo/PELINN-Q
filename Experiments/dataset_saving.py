from qiskit.quantum_info import SparsePauliOp
from pelinn.data.dataset_generator import GenerateQuantumDataset

observables = [SparsePauliOp.from_list([("ZIZ", 1.0)])]

generator = GenerateQuantumDataset(
    n_qubits=3,
    depth=4,
    circuit_type="variational",
    noise_config={"noise_list": []},
    observables=observables,
    circuit_params={"n_terms": 3},
    shots=1024,
    seed=123,
)

dataset = generator.generate_dataset(n_samples=150)
if len(dataset.dataset) == 0:
    raise RuntimeError("Dataset generation returned 0 samples. Check circuit_params/noise settings.")
dataset.save_dataset("data/demo_dataset.npz", format="numpy")
print("Saved: data/demo_dataset.npz")
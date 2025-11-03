# Classes in `pelinn/data/qiskit_dataset.py`

## `Sample`
A `@dataclass` representing a single data point for quantum error mitigation.

### Fields
- **`x`** (`numpy.ndarray`): composite feature vector containing circuit
  statistics, noise parameters, and measurement outcomes.
- **`y_noisy`** (`float`): measured expectation value under a noisy backend.
- **`y_ideal`** (`float`): reference expectation value from an ideal simulation.
- **`meta`** (`Dict`): additional metadata such as the original circuit, the
  noise configuration used, and the number of shots.

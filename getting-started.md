---
layout: default
title: Getting Started
---

# Getting Started with PELINN-Q

This guide helps you quickly get started with PELINN-Q.

## Step 1: Installation

### Clone the Repository

```bash
git clone https://github.com/BramDo/PELINN-Q.git
cd PELINN-Q
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install PELINN-Q

```bash
pip install -e .
```

## Step 2: Your First Experiment

### Basic Example

Create a new Python file `my_first_experiment.py`:

```python
import numpy as np
from pelinn import LiquidNeuralNetwork

# Create dummy data
X_train = np.random.randn(100, 10)
y_train = np.random.randn(100, 5)

# Initialize the network
lnn = LiquidNeuralNetwork(
    input_size=10,
    hidden_size=20,
    output_size=5
)

# Train
lnn.train(X_train, y_train, epochs=50)

# Test
X_test = np.random.randn(10, 10)
predictions = lnn.predict(X_test)

print("Predictions:", predictions)
```

Run the script:

```bash
python my_first_experiment.py
```

## Step 3: Quantum Error Mitigation

### Quantum Circuit with Error Mitigation

```python
from qiskit import QuantumCircuit, execute, Aer
from pelinn.qem import QuantumErrorMitigator
from pelinn import LiquidNeuralNetwork

# Create a quantum circuit
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

# Simulate with noise
backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=1000)
noisy_results = job.result().get_counts()

# Train a mitigator
lnn = LiquidNeuralNetwork(input_size=4, hidden_size=10, output_size=4)
mitigator = QuantumErrorMitigator(lnn)

# Apply mitigation
mitigated_results = mitigator.mitigate(noisy_results)

print("Noisy results:", noisy_results)
print("Mitigated results:", mitigated_results)
```

## Step 4: Use Existing Scripts

PELINN-Q contains example scripts in the `scripts/` folder:

```bash
# Run an example script
python scripts/example.py
```

## Step 5: Experiment with Notebooks

Open the Jupyter notebooks in the `notebooks/` folder:

```bash
jupyter notebook notebooks/
```

These notebooks contain:
- Detailed tutorials
- Visualizations
- Comparisons with other methods

## Customize Configuration

Customize the configuration via `test.ini`:

```ini
[model]
input_size = 10
hidden_size = 20
output_size = 5

[training]
epochs = 100
learning_rate = 0.001
```

## Next Steps

- Check out the [full documentation](./documentation) for more details
- Read more [about the project](./about)
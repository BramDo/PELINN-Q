---
layout: default
title: Documentation
---

# Documentation

## Overview

PELINN-Q provides tools for quantum error mitigation with liquid neural networks. This documentation helps you use and understand the library.

---

## 📖 Class Explainers

Detailed explanations of the key classes and architectures in PELINN-Q:

### [Understanding LTCCell](./docs/explain-model)
In-depth explanation of the **Liquid Time-Constant (LTC)** recurrent neural network cell. Learn about:
- Weights and learnable parameters (time-constant pathway, gating pathway, attractor vector)
- Forward dynamics and integration with Euler steps
- Regularization hooks for training

---

## 🚀 Installation

### Requirements

- Python 3.8+
- NumPy
- PyTorch
- Qiskit (for quantum simulations)

### Install

```bash
# Clone the repository
git clone https://github.com/BramDo/PELINN-Q.git
cd PELINN-Q
```

---

### Train a Liquid Neural Network

```python
from pelinn import LiquidNeuralNetwork
from pelinn.qem import QuantumErrorMitigator

# Initialize the network
lnn = LiquidNeuralNetwork(
    input_size=10,
    hidden_size=20,
    output_size=5
)

# Train the model
lnn.train(training_data, epochs=100)
```

### Apply Quantum Error Mitigation

```python
# Create a QEM mitigator
mitigator = QuantumErrorMitigator(lnn)

# Apply mitigation to quantum results
mitigated_results = mitigator.mitigate(raw_quantum_data)
```

---

## 📚 API Referentie

### LiquidNeuralNetwork

The main class for liquid neural networks.

**Parameters:**
- `input_size` (int): Number of input features
- `hidden_size` (int): Number of hidden neurons
- `output_size` (int): Number of output features
- `tau` (float): Time constant for the network

**Methods:**
- `train(data, epochs)`: Train the network
- `predict(input)`: Predict the output for a given input

### QuantumErrorMitigator

Class for quantum error mitigation.

**Parameters:**
- `model`: Trained liquid neural network model

**Methods:**
- `mitigate(quantum_data)`: Apply error mitigation
- `evaluate(test_data)`: Evaluate mitigation performance

---

## ⚙️ Configuration

You can configure PELINN-Q through a `config.ini` file:

```ini
[model]
input_size = 10
hidden_size = 20
output_size = 5
learning_rate = 0.001

[training]
epochs = 100
batch_size = 32
```

---

## 📓 Voorbeelden

Check the `notebooks/` folder for Jupyter Notebook examples and the `scripts/` folder for standalone scripts.

---

## 🔧 Troubleshooting

### Common issues

**ImportError**: Make sure all dependencies are installed
```bash
pip install -r requirements.txt
```

**CUDA errors**: Check that PyTorch is installed correctly for your system

---

## 🔗 More Information

- [Getting Started Guide](./getting-started)
- [About PELINN-Q](./about)
- [GitHub Repository](https://github.com/BramDo/PELINN-Q)

[Back to Home](./)

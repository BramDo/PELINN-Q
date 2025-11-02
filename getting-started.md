---
layout: default
title: Getting Started
---

# Getting Started met PELINN-Q

Deze gids helpt je om snel aan de slag te gaan met PELINN-Q.

## Stap 1: Installatie

### Clone de Repository

```bash
git clone https://github.com/BramDo/PELINN-Q.git
cd PELINN-Q
```

### Installeer Dependencies

```bash
pip install -r requirements.txt
```

### Installeer PELINN-Q

```bash
pip install -e .
```

## Stap 2: Je Eerste Experiment

### Basis Voorbeeld

Maak een nieuw Python bestand `my_first_experiment.py`:

```python
import numpy as np
from pelinn import LiquidNeuralNetwork

# Maak dummy data
X_train = np.random.randn(100, 10)
y_train = np.random.randn(100, 5)

# Initialiseer het netwerk
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

Run het script:

```bash
python my_first_experiment.py
```

## Stap 3: Quantum Error Mitigation

### Quantum Circuit met Error Mitigation

```python
from qiskit import QuantumCircuit, execute, Aer
from pelinn.qem import QuantumErrorMitigator
from pelinn import LiquidNeuralNetwork

# Maak een quantum circuit
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

# Simuleer met ruis
backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=1000)
noisy_results = job.result().get_counts()

# Train een mitigator
lnn = LiquidNeuralNetwork(input_size=4, hidden_size=10, output_size=4)
mitigator = QuantumErrorMitigator(lnn)

# Pas mitigation toe
mitigated_results = mitigator.mitigate(noisy_results)

print("Noisy results:", noisy_results)
print("Mitigated results:", mitigated_results)
```

## Stap 4: Gebruik Bestaande Scripts

PELINN-Q bevat voorbeeldscripts in de `scripts/` folder:

```bash
# Run een voorbeeld script
python scripts/example.py
```

## Stap 5: Experimenteer met Notebooks

Open de Jupyter notebooks in de `notebooks/` folder:

```bash
jupyter notebook notebooks/
```

Deze notebooks bevatten:
- Gedetailleerde tutorials
- Visualisaties
- Vergelijkingen met andere methoden

## Configuratie Aanpassen

Pas de configuratie aan via `test.ini`:

```ini
[model]
input_size = 10
hidden_size = 20
output_size = 5

[training]
epochs = 100
learning_rate = 0.001
```

## Volgende Stappen

- Bekijk de [volledige documentatie](./documentation) voor meer details
- Lees meer [over het project](./about)
- Experimenteer met eigen quantum circuits

## Hulp Nodig?

- Open een [GitHub Issue](https://github.com/BramDo/PELINN-Q/issues)
- Bekijk de [documentatie](./documentation)

[Terug naar Home](./)
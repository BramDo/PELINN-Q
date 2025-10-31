# PE-LiNN-Q

Liquid neural networks for quantum error mitigation (QEM). This repository houses the PyTorch implementation of the PE-LiNN regressor together with utilities to synthesise noisy/ideal datasets from Qiskit, reproduce baseline mitigators built on Mitiq, and run end-to-end benchmarks.

## Repository Layout
- `pelinn/` – core implementation of the liquid time-constant model (`PELiNNQEM`) and Qiskit-driven data synthesis helpers.
- `scripts/` – runnable entry points; `eval_benchmarks.py` trains/evaluates PE-LiNN alongside classical mitigation baselines, while `train_pelinn.py` sketches a minimal training loop.
- `baselines/` – wrappers for Mitiq-based zero-noise extrapolation (ZNE) and Clifford data regression (CDR).
- `notebooks/` – exploratory demos illustrating the math and usage of the architecture.
- `test.ini` – sample command line plus expected log output from a benchmark run.

## Prerequisites
The code targets Python 3.10+ with recent releases of PyTorch, Qiskit, and Mitiq. A working CUDA toolchain is optional but recommended for faster training.

Minimum pip environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch numpy qiskit qiskit-aer mitiq
```

If you only need the PE-LiNN model without baseline comparisons you may omit `mitiq`. In that case, baseline scripts will raise a runtime warning and be skipped.

## Quick Start
The benchmark driver (`scripts/eval_benchmarks.py`) synthesises noisy and ideal expectation values, trains PE-LiNN, and compares it to baseline mitigators. Display the available options with:

```bash
python scripts/eval_benchmarks.py --help
```

A representative CPU-only run mirroring `test.ini`:

```bash
python scripts/eval_benchmarks.py \
    --num-circuits 80 \
    --test-fraction 0.2 \
    --baselines cdr \
    --device cpu \
    --hid-dim 96 \
    --steps 4 \
    --dt 0.1 \
    --epochs 200 \
    --batch-size 8 \
    --lr 5e-5 \
    --alpha-inv 0.3 \
    --loss huber \
    --huber-beta 0.15 \
    --tanh-head \
    --clamp-outputs 0.5 \
    --outlier-z 1 \
    --preview-samples 30
```

> Expect minutes on a laptop CPU; GPU execution is controlled via `--device` (falls back to CPU if CUDA is unavailable).

Key outputs include per-method regression metrics (MAE, RMSE, max error) and optional sample-by-sample predictions.

## Data Generation Pipeline
`pelinn/data/qiskit_dataset.py` provides:
- `circuit_features`: simple circuit descriptors (qubit count, depth, gate tallies).
- `make_noise_model`: configurable mix of depolarising, amplitude damping, and readout noise via Qiskit Aer.
- `synthesize_samples`: pairs noisy and ideal expectation values for a grid of circuits, observables, and noise settings; returns structured `Sample` objects with metadata.

These utilities are reusable if you want to plug PE-LiNN into different circuit families or noise regimes. Extend the feature vector by editing `Sample.x` construction before training.

## PE-LiNN Model
`pelinn/model.py` implements:
- `LTCCell`: a liquid time-constant recurrent cell with cached regularisers.
- `PELiNNQEM`: wraps the cell, exposes `compute_loss` (MSE/Huber + invariance penalties + optional regularisation), and supports configurable integration steps and tanh output head.
- `physics_loss`: standalone loss helper if you prefer to integrate PE-LiNN into custom loops.

The lightweight reference loop in `scripts/train_pelinn.py` shows how to feed `Sample` objects into PyTorch’s `DataLoader`, form per-circuit invariance groups, and optimise with AdamW.

## Baselines
Baseline wrappers live in `baselines/` and expect Mitiq ≥ 3.0:
- `zne_mitiq.py` – global-folding zero-noise extrapolation with configurable scale factors and extrapolators.
- `cdr_mitiq.py` – Clifford data regression with optional custom training circuit generators.

`eval_benchmarks.py` dispatches to these modules via `--baselines zne,cdr`. Additional baselines (e.g., probabilistic error cancellation) can be added by following the same pattern and updating `resolve_baseline`.

## Notebooks
The `notebooks/` directory collects Jupyter demos:
- `demo.ipynb` – end-to-end walkthrough of dataset synthesis and PE-LiNN training.
- `demo_math.ipynb` plus PDFs – mathematical background on liquid time-constant networks and their relation to quantum error mitigation.

Launch them with JupyterLab or VS Code after activating your environment.

## Troubleshooting Tips
- Qiskit Aer or Mitiq import errors: verify the environment matches the prerequisite versions and that `python -m pip install qiskit-aer mitiq` succeeded.
- Long runtimes: reduce `--num-circuits`, `--epochs`, or `--max-eval-samples` while iterating.


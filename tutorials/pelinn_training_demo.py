"""
Minimal training demo: generate a noisy/noiseless dataset with GenerateQuantumDataset
and fit the PELiNNQEM LTC model on it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from qiskit.quantum_info import SparsePauliOp

# Ensure local modules (src/, pelinn/) are importable when running from repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_generator import GenerateQuantumDataset
from src.model import PELiNNQEM


def build_dataset(n_samples: int = 200, seed: int | None = 42):
    """Generate a small dataset: single-qubit, one observable (Z), simple depolarizing noise."""
    observables = [SparsePauliOp("Z")]  # n_observables = 1 → matches PELiNN head
    noise_cfg = {"noise_list": [{"type": "depolarizing", "p": 0.02}]}

    gen = GenerateQuantumDataset(
        n_qubits=1,
        depth=2,
        circuit_type="random",
        noise_config=noise_cfg,
        observables=observables,
        shots=1024,
        seed=seed,
        save_circuits=False,
        opt_level=1,
    )
    dataset = gen.generate_dataset(n_samples=n_samples, show_progress=True)
    return dataset.dataset_to_torch()


def train_model(
    X: torch.Tensor,
    Y: torch.Tensor,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
):
    """Train PELiNNQEM on the generated dataset and print losses."""
    ds = TensorDataset(X, Y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = PELiNNQEM(in_dim=X.shape[1], hid_dim=96, steps=6, use_tanh_head=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        running = 0.0
        for xb, yb in dl:
            target = yb.squeeze(-1)  # Y is (batch, 1) for single observable
            pred = model(xb)
            loss = model.compute_loss(pred, target, groups=None, alpha_inv=0.1, loss_type="mse")
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        print(f"epoch {epoch:02d} | loss {running / len(ds):.4f}")

    return model


def main():
    X, Y = build_dataset()
    model = train_model(X, Y)

    # Quick sanity check on a few samples
    with torch.no_grad():
        preds = model(X[:5])
        print("sample preds:", preds.tolist())
        print("sample targets:", Y[:5].squeeze(-1).tolist())


if __name__ == "__main__":
    main()

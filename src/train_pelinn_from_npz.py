#!/usr/bin/env python3
"""Train the PE-LiNN model using an NPZ dataset compatible with GenerateQuantumDataset."""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pelinn.model import PELiNNQEM, physics_loss  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="data/demo_dataset.npz", help="Path to NPZ dataset containing arrays X (features), Y (targets), metadata.")
    parser.add_argument("--output-dir", type=str, default="artifacts_npz", help="Directory to store plots and summary.")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for AdamW.")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay for AdamW.")
    parser.add_argument("--hid-dim", type=int, default=96, help="Hidden dimension of LTCCell.")
    parser.add_argument("--steps", type=int, default=6, help="Recurrent integration steps inside LTCCell.")
    parser.add_argument("--dt", type=float, default=0.25, help="Integration step size dt in LTCCell.")
    parser.add_argument("--tanh-head", action="store_true", help="Clamp outputs via tanh head.")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction (0 => no validation).")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"), help="Logging level.")
    parser.add_argument("--no-normalise", action="store_true", help="Disable feature normalisation.")
    return parser.parse_args(argv)


class NpzDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        circuit_ids: Iterable[int],
        mean: np.ndarray | None,
        std: np.ndarray | None,
        normalise: bool,
    ) -> None:
        if normalise:
            if mean is None:
                mean = features.mean(axis=0)
            if std is None:
                std = features.std(axis=0)
            std = np.maximum(std, 1e-6)
            self.feature_mean = mean.astype(np.float32)
            self.feature_std = std.astype(np.float32)
            features = (features - self.feature_mean) / self.feature_std
        else:
            self.feature_mean = mean.astype(np.float32) if mean is not None else np.zeros(features.shape[1], dtype=np.float32)
            self.feature_std = std.astype(np.float32) if std is not None else np.ones(features.shape[1], dtype=np.float32)
        self.X = features.astype(np.float32)
        self.y = targets.astype(np.float32)
        self.cid = np.array(list(circuit_ids), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.y[idx]),
            torch.tensor(self.cid[idx]),
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(num_samples: int, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(num_samples)
    if val_fraction <= 0:
        return indices, np.empty(0, dtype=int)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    n_val = int(round(num_samples * val_fraction))
    if 0 < num_samples <= 1:
        n_val = 0
    n_val = min(max(n_val, 1), num_samples - 1) if num_samples > 1 else 0
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    if train_idx.size == 0:
        raise ValueError("Training split empty; adjust --val-fraction.")
    return train_idx, val_idx


def load_npz_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    blob = np.load(path, allow_pickle=True)
    if "X" not in blob or "Y" not in blob:
        raise ValueError("NPZ file must contain 'X' (features) and 'Y' (targets) arrays.")
    X = np.asarray(blob["X"], dtype=np.float32)
    Y = np.asarray(blob["Y"], dtype=np.float32)
    if Y.ndim == 2 and Y.shape[1] > 1:
        logging.warning("Multiple target columns found (%d); using first column for training.", Y.shape[1])
        Y = Y[:, 0]
    elif Y.ndim > 1:
        Y = Y.reshape(-1)
    else:
        Y = Y.astype(np.float32)
    metadata = blob.get("metadata")
    if isinstance(metadata, np.ndarray) and metadata.shape == ():
        metadata = metadata.item()
    circuit_ids = np.arange(len(X))
    if metadata is not None:
        try:
            if isinstance(metadata, dict) and "circuit_index" in metadata:
                circuit_ids = np.array(metadata["circuit_index"], dtype=int)
            elif isinstance(metadata, (list, tuple, np.ndarray)):
                circuit_ids = np.array([(_extract_circuit_idx(entry, idx)) for idx, entry in enumerate(metadata)], dtype=int)
        except Exception as exc:
            logging.warning("Failed to parse circuit indices from metadata (%s); using sequential ids.", exc)
    logging.debug("Loaded dataset: X%s, Y%s", X.shape, Y.shape)
    return X, Y, circuit_ids


def _extract_circuit_idx(entry: object, default: int) -> int:
    if isinstance(entry, dict) and "circuit_index" in entry:
        return int(entry["circuit_index"])
    if isinstance(entry, dict) and "sample_index" in entry:
        return int(entry["sample_index"])
    return default


def make_groups(ids: Sequence[int]) -> List[List[int]]:
    grouped: Dict[int, List[int]] = {}
    for idx, cid in enumerate(ids):
        grouped.setdefault(int(cid), []).append(idx)
    return list(grouped.values())


def train_epoch(
    model: PELiNNQEM,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    alpha_inv: float,
) -> float:
    model.train()
    running = 0.0
    batches = 0
    for batch_idx, (X, y, cid) in enumerate(loader, start=1):
        X = X.to(device)
        y = y.to(device)
        ids = cid.tolist()
        preds = model(X)
        loss = physics_loss(
            preds,
            y,
            make_groups(ids),
            alpha_inv=alpha_inv,
            reg_gate=model.last_gate_reg,
            reg_A=model.last_A_reg,
        )
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        loss_val = float(loss.detach())
        running += loss_val
        batches += 1
        logging.debug("Batch %d | loss=%.6f", batch_idx, loss_val)
    return running / max(batches, 1)


def evaluate(model: PELiNNQEM, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    preds: List[np.ndarray] = []
    targs: List[np.ndarray] = []
    with torch.no_grad():
        for X, y, _ in loader:
            preds.append(model(X.to(device)).cpu().numpy())
            targs.append(y.numpy())
    if not preds:
        return {"mae": float("nan"), "rmse": float("nan")}
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targs)
    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    logging.debug("Eval | mae=%.6f rmse=%.6f", mae, rmse)
    return {"mae": mae, "rmse": rmse, "y_pred": y_pred, "y_true": y_true}


def plot_metrics(training_losses: List[float], val_losses: List[float], y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(training_losses) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, training_losses, label="train_loss")
    if val_losses:
        plt.plot(epochs, val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    loss_path = out_dir / "loss_curve.png"
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()
    logging.info("Saved loss curve to %s", loss_path)

    if y_true.size and y_pred.size:
        plt.figure(figsize=(5, 5))
        plt.scatter(y_true, y_pred, alpha=0.6, edgecolor="k")
        min_y = float(min(y_true.min(), y_pred.min()))
        max_y = float(max(y_true.max(), y_pred.max()))
        plt.plot([min_y, max_y], [min_y, max_y], "r--", label="ideal")
        plt.xlabel("True y")
        plt.ylabel("Predicted y")
        plt.title("Prediction vs True (validation)")
        plt.legend()
        scatter_path = out_dir / "pred_vs_true.png"
        plt.tight_layout()
        plt.savefig(scatter_path, dpi=150)
        plt.close()
        logging.info("Saved prediction scatter plot to %s", scatter_path)


def write_summary(
    summary_path: Path,
    model: PELiNNQEM,
    metrics: Dict[str, float],
    config: argparse.Namespace,
    dataset_info: Dict[str, float],
) -> None:
    input_dim = getattr(model.cell.W_tx, "in_features", None)
    hidden_dim = getattr(model.cell.W_tx, "out_features", None)
    summary = {
        "model": "PELiNNQEM",
        "configuration": {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "steps": model.steps,
            "dt": model.dt,
            "use_tanh_head": bool(model.use_tanh_head),
        },
        "training_parameters": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.lr,
            "weight_decay": config.weight_decay,
            "alpha_inv": 0.1,
            "normalised": not config.no_normalise,
        },
        "dataset": dataset_info,
        "metrics": {k: float(v) for k, v in metrics.items() if k not in {"y_pred", "y_true"}},
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    logging.info("Wrote training summary to %s", summary_path)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    set_seed(args.seed)

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = (ROOT / dataset_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    logging.info("Loading NPZ dataset from %s", dataset_path)

    X, Y, circuit_ids = load_npz_dataset(dataset_path)
    train_idx, val_idx = split_indices(len(X), args.val_fraction, args.seed)
    logging.info("Train samples: %d | Validation samples: %d", train_idx.size, val_idx.size)

    normalise = not args.no_normalise
    train_dataset = NpzDataset(X[train_idx], Y[train_idx], circuit_ids[train_idx], mean=None, std=None, normalise=normalise)
    val_dataset = None
    if val_idx.size:
        val_dataset = NpzDataset(
            X[val_idx],
            Y[val_idx],
            circuit_ids[val_idx],
            mean=train_dataset.feature_mean,
            std=train_dataset.feature_std,
            normalise=normalise,
        )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) if val_dataset else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    model = PELiNNQEM(
        in_dim=train_dataset.X.shape[1],
        hid_dim=args.hid_dim,
        steps=args.steps,
        dt=args.dt,
        use_tanh_head=args.tanh_head,
    ).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimiser, device, alpha_inv=0.1)
        train_losses.append(train_loss)
        log_msg = f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.6f}"
        if val_loader:
            metrics = evaluate(model, val_loader, device)
            val_losses.append(metrics["rmse"])
            log_msg += f" | val_mae={metrics['mae']:.6f} val_rmse={metrics['rmse']:.6f}"
        logging.info(log_msg)

    final_metrics = {"mae": float("nan"), "rmse": float(train_losses[-1]) if train_losses else float("nan")}
    y_true = np.array([])
    y_pred = np.array([])

    if val_loader:
        metrics = evaluate(model, val_loader, device)
        final_metrics.update({"mae": metrics["mae"], "rmse": metrics["rmse"]})
        y_true = metrics["y_true"]
        y_pred = metrics["y_pred"]

    output_dir = Path(args.output_dir).resolve()
    plot_metrics(train_losses, val_losses, y_true, y_pred, output_dir)

    dataset_info = {
        "dataset_path": str(dataset_path),
        "num_train_samples": int(train_idx.size),
        "num_val_samples": int(val_idx.size),
        "num_features": int(train_dataset.X.shape[1]),
        "normalised": bool(normalise),
    }
    summary_path = output_dir / "training_summary.json"
    write_summary(summary_path, model, final_metrics, args, dataset_info)


if __name__ == "__main__":
    main()

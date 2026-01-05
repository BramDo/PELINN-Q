#!/usr/bin/env python3
"""Train the PE-LiNN QEM model on synthesized or saved datasets."""

import argparse
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Ensure the repository root is on sys.path so `pelinn` imports resolve when the
# script is launched directly (python scripts/train_pelinn.py).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pelinn.model import PELiNNQEM, physics_loss

try:  # Prefer the canonical Sample dataclass when Qiskit is installed.
    from pelinn.data.qiskit_dataset import Sample, synthesize_samples  # type: ignore
except Exception:  # pragma: no cover - fallback for minimal envs
    synthesize_samples = None

    @dataclass
    class Sample:  # type: ignore
        x: np.ndarray
        y_noisy: float
        y_ideal: float
        meta: Dict


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default=None, help="Path to a saved dataset (.npz or .pt).")
    parser.add_argument(
        "--dataset-format",
        type=str,
        default="auto",
        choices=("auto", "npz", "pt"),
        help="Dataset file format (auto-detected when possible).",
    )
    parser.add_argument(
        "--save-samples",
        type=str,
        default=None,
        help="Optional path to store synthesized samples as .npz (features, labels, metadata).",
    )
    parser.add_argument("--num-qubits", type=int, default=4, help="Number of qubits per synthesized circuit.")
    parser.add_argument("--num-circuits", type=int, default=32, help="Number of base circuits to synthesize.")
    parser.add_argument("--shots-noisy", type=int, default=4096, help="Shot count for noisy estimation.")
    parser.add_argument("--shots-ideal", type=int, default=0, help="Shot count for ideal estimation (0 => exact).")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Fraction of circuits reserved for validation.")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay.")
    parser.add_argument("--alpha-inv", type=float, default=0.1, help="Physics invariance penalty weight.")
    parser.add_argument("--hid-dim", type=int, default=96, help="Hidden dimension of the PE-LiNN model.")
    parser.add_argument("--steps", type=int, default=6, help="Number of recurrent integration steps.")
    parser.add_argument("--dt", type=float, default=0.25, help="Integration step size for the liquid cell.")
    parser.add_argument("--tanh-head", action="store_true", help="Apply tanh head to bound outputs to [-1, 1].")
    parser.add_argument("--seed", type=int, default=1234, help="Global RNG seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Preferred torch device (falls back to cpu).")
    parser.add_argument("--no-normalise", action="store_true", help="Disable feature normalisation.")
    parser.add_argument("--save-model", type=str, default=None, help="Optional path to persist the trained model state.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging verbosity.",
    )
    return parser.parse_args(argv)


class QemDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], mean: np.ndarray | None = None, std: np.ndarray | None = None, normalise: bool = True):
        features = np.stack([np.asarray(s.x, dtype=np.float32) for s in samples])
        targets = np.array([float(s.y_ideal) for s in samples], dtype=np.float32)
        cids: List[int] = []
        for idx, sample in enumerate(samples):
            meta = getattr(sample, "meta", {}) or {}
            cid_val = meta.get("circuit_index")
            if cid_val is None:
                qc_obj = meta.get("qc")
                cid_val = id(qc_obj) if qc_obj is not None else idx
            cids.append(int(cid_val))

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
        self.y = targets
        self.cid = np.array(cids, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.cid[idx]


def make_groups(cids: Iterable[int]) -> List[List[int]]:
    grouped: Dict[int, List[int]] = {}
    for idx, cid in enumerate(cids):
        grouped.setdefault(int(cid), []).append(idx)
    return list(grouped.values())


def train_epoch(model: PELiNNQEM, loader: DataLoader, optimiser: torch.optim.Optimizer, device: torch.device, alpha_inv: float) -> float:
    model.train()
    running = 0.0
    batches = 0
    for X, y, cid in loader:
        X = X.to(device)
        y = y.to(device)
        cid_list = cid.tolist()
        preds = model(X)
        loss = physics_loss(
            preds,
            y,
            make_groups(cid_list),
            alpha_inv=alpha_inv,
            reg_gate=model.last_gate_reg,
            reg_A=model.last_A_reg,
        )
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        running += float(loss.detach())
        batches += 1
    return running / max(1, batches)


def evaluate(model: PELiNNQEM, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    with torch.no_grad():
        for X, y, _ in loader:
            out = model(X.to(device)).cpu().numpy()
            preds.append(out)
            targets.append(y.cpu().numpy())
    if not preds:
        return {"mae": float("nan"), "rmse": float("nan")}
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)
    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    return {"mae": mae, "rmse": rmse}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_circuit_family(num_qubits: int, num_circuits: int, rng: random.Random):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp

    circuits: List[QuantumCircuit] = []
    observables: List[SparsePauliOp] = []
    for _ in range(num_circuits):
        qc = QuantumCircuit(num_qubits)
        for qubit in range(num_qubits):
            qc.h(qubit)
        for layer in range(4):
            for qubit in range(num_qubits):
                qc.rz(rng.uniform(-np.pi, np.pi), qubit)
            for qubit in range(num_qubits - 1):
                qc.cx(qubit, qubit + 1)
            target = rng.randrange(num_qubits)
            qc.rx(rng.uniform(-np.pi, np.pi), target)
        qc.measure_all(False)
        circuits.append(qc)
        observables.append(SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1.0)]))
    return circuits, observables


def default_noise_grid() -> List[Dict[str, float]]:
    return [
        {"p1_depol": 0.02, "p2_depol": 0.08, "p_amp": 0.02, "readout_p01": 0.08, "readout_p10": 0.08},
        {"p1_depol": 0.04, "p2_depol": 0.12, "p_amp": 0.03, "readout_p01": 0.12, "readout_p10": 0.12},
    ]


def split_samples(samples: Sequence[Sample], val_fraction: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    if val_fraction <= 0.0:
        return list(samples), []
    by_circuit: Dict[int, List[Sample]] = {}
    for idx, sample in enumerate(samples):
        meta = getattr(sample, "meta", {}) or {}
        cid = meta.get("circuit_index")
        if cid is None:
            cid = idx
        by_circuit.setdefault(int(cid), []).append(sample)

    circuit_ids = list(by_circuit.keys())
    rng = random.Random(seed)
    rng.shuffle(circuit_ids)
    n_val = max(1, int(round(len(circuit_ids) * val_fraction))) if val_fraction < 1.0 else len(circuit_ids)
    val_ids = set(circuit_ids[:n_val])
    train, val = [], []
    for cid, group in by_circuit.items():
        (val if cid in val_ids else train).extend(group)
    if not train:
        raise ValueError("Training split is empty; adjust --val-fraction.")
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def load_samples_from_npz(path: Path) -> List[Sample]:
    blob = np.load(path, allow_pickle=True)
    X = np.asarray(blob["X"], dtype=np.float32)
    Y = np.asarray(blob["Y"], dtype=np.float32)
    metadata = blob.get("metadata")
    samples: List[Sample] = []
    for idx, (x_row, y_val) in enumerate(zip(X, Y)):
        x_arr = np.asarray(x_row, dtype=np.float32).reshape(-1)
        y_scalar = _scalarise_target(y_val, idx)
        meta = _extract_metadata(metadata, idx)
        samples.append(
            Sample(
                x=x_arr,
                y_noisy=float(meta.get("y_noisy", np.nan)) if "y_noisy" in meta else float("nan"),
                y_ideal=y_scalar,
                meta=meta,
            )
        )
    return samples


def load_samples_from_pt(path: Path) -> List[Sample]:
    blob = torch.load(path, map_location="cpu")
    X = np.asarray(blob["X"].cpu().numpy(), dtype=np.float32)
    Y = np.asarray(blob["Y"].cpu().numpy(), dtype=np.float32)
    metadata = blob.get("metadata")
    samples: List[Sample] = []
    for idx, (x_row, y_val) in enumerate(zip(X, Y)):
        x_arr = np.asarray(x_row, dtype=np.float32).reshape(-1)
        y_scalar = _scalarise_target(y_val, idx)
        meta = _extract_metadata(metadata, idx)
        samples.append(
            Sample(
                x=x_arr,
                y_noisy=float(meta.get("y_noisy", np.nan)) if isinstance(meta, dict) and "y_noisy" in meta else float("nan"),
                y_ideal=y_scalar,
                meta=meta,
            )
        )
    return samples


def _extract_metadata(metadata_obj, idx: int) -> Dict:
    if metadata_obj is None:
        return {"circuit_index": idx}
    if isinstance(metadata_obj, dict):
        meta = dict(metadata_obj)
        meta.setdefault("circuit_index", idx)
        meta["sample_index"] = idx
        return meta
    if isinstance(metadata_obj, (list, tuple)) and len(metadata_obj) == 0:
        return {"circuit_index": idx}
    if isinstance(metadata_obj, (list, tuple)) and idx < len(metadata_obj):
        entry = metadata_obj[idx]
        if isinstance(entry, dict):
            meta = dict(entry)
        else:
            meta = {"value": entry}
        meta.setdefault("circuit_index", meta.get("circuit_index", idx))
        return meta
    if isinstance(metadata_obj, np.ndarray):
        if metadata_obj.dtype == object and metadata_obj.ndim == 0:
            return _extract_metadata(metadata_obj.item(), idx)
        if metadata_obj.dtype == object and idx < len(metadata_obj):
            entry = metadata_obj[idx]
            if isinstance(entry, dict):
                meta = dict(entry)
            else:
                meta = {"value": entry}
            meta.setdefault("circuit_index", meta.get("circuit_index", idx))
            return meta
    return {"circuit_index": idx}


def _scalarise_target(value: np.ndarray | Sequence[float] | float, idx: int) -> float:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        return float(arr)
    flat = arr.reshape(-1)
    if flat.size == 0:
        raise ValueError(f"Sample {idx} has empty target array.")
    if flat.size > 1:
        logging.warning(
            "Sample %d target has %d observables; using the first entry. Adjust dataset or loader if needed.",
            idx,
            flat.size,
        )
    return float(flat[0])


def save_samples_npz(samples: Sequence[Sample], path: Path) -> None:
    X = np.stack([np.asarray(s.x, dtype=np.float32) for s in samples])
    Y = np.array([float(s.y_ideal) for s in samples], dtype=np.float32)
    enriched_meta = []
    for sample in samples:
        meta_dict = dict(getattr(sample, "meta", {}) or {})
        meta_dict.setdefault("circuit_index", meta_dict.get("circuit_index"))
        meta_dict["y_noisy"] = float(getattr(sample, "y_noisy", float("nan")))
        enriched_meta.append(meta_dict)
    metadata = np.array(enriched_meta, dtype=object)
    np.savez(path, X=X, Y=Y, metadata=metadata)


def auto_detect_format(path: Path, explicit: str) -> str:
    if explicit != "auto":
        return explicit
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return "npz"
    if suffix == ".pt":
        return "pt"
    raise ValueError(f"Unable to auto-detect dataset format from suffix '{suffix}'. Specify --dataset-format.")


def load_samples(path: Path, fmt: str) -> List[Sample]:
    if fmt == "npz":
        return load_samples_from_npz(path)
    if fmt == "pt":
        return load_samples_from_pt(path)
    raise ValueError(f"Unsupported dataset format '{fmt}'.")


def ensure_samples(args: argparse.Namespace, rng: random.Random) -> List[Sample]:
    if args.dataset:
        dataset_path = Path(args.dataset).resolve()
        fmt = auto_detect_format(dataset_path, args.dataset_format)
        logging.info("Loading dataset from %s (%s)", dataset_path, fmt)
        samples = load_samples(dataset_path, fmt)
        logging.info("Loaded %d samples.", len(samples))
        return samples

    if synthesize_samples is None:
        logging.warning("Qiskit primitives unavailable; falling back to synthetic Gaussian demo.")
        samples: List[Sample] = []
        in_dim = 10
        for idx in range(1000):
            x = np.random.randn(in_dim).astype(np.float32)
            y = float(np.tanh(0.8 * x[0] + 0.4 * x[1] - 0.2 * x[2]))
            samples.append(Sample(x=x, y_noisy=float("nan"), y_ideal=y, meta={"circuit_index": idx % 50}))
        return samples

    logging.info("Synthesizing samples with Qiskit (num_qubits=%d, num_circuits=%d).", args.num_qubits, args.num_circuits)
    circuits, observables = make_circuit_family(args.num_qubits, args.num_circuits, rng)
    samples = synthesize_samples(
        circuits,
        observables,
        default_noise_grid(),
        shots_noisy=args.shots_noisy,
        shots_ideal=args.shots_ideal,
    )
    logging.info("Synthesized %d samples.", len(samples))
    if args.save_samples:
        out_path = Path(args.save_samples).resolve()
        save_samples_npz(samples, out_path)
        logging.info("Saved synthesized samples to %s", out_path)
    return samples


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    # Tone down verbose transpiler logs when Qiskit is present.
    logging.getLogger("qiskit").setLevel(logging.WARNING)
    logging.getLogger("qiskit.transpiler").setLevel(logging.WARNING)

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    logging.info("Using device: %s", device)

    rng = random.Random(args.seed)
    samples = ensure_samples(args, rng)

    train_samples, val_samples = split_samples(samples, args.val_fraction, args.seed)
    normalise = not args.no_normalise
    train_dataset = QemDataset(train_samples, normalise=normalise)
    val_dataset = QemDataset(
        val_samples,
        mean=train_dataset.feature_mean,
        std=train_dataset.feature_std,
        normalise=normalise,
    ) if val_samples else None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) if val_dataset else None

    model = PELiNNQEM(
        in_dim=train_dataset.X.shape[1],
        hid_dim=args.hid_dim,
        steps=args.steps,
        dt=args.dt,
        use_tanh_head=args.tanh_head,
    ).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimiser, device, alpha_inv=args.alpha_inv)
        if val_loader:
            metrics = evaluate(model, val_loader, device)
            logging.info("Epoch %d/%d | loss=%.5f | val_mae=%.5f | val_rmse=%.5f", epoch, args.epochs, loss, metrics["mae"], metrics["rmse"])
        else:
            logging.info("Epoch %d/%d | loss=%.5f", epoch, args.epochs, loss)

    if args.save_model:
        out_path = Path(args.save_model).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "feature_mean": train_dataset.feature_mean,
            "feature_std": train_dataset.feature_std,
            "config": vars(args),
        }, out_path)
        logging.info("Saved model checkpoint to %s", out_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train the PE-LiNN model from an NPZ dataset, with optional Qiskit synthesis and baselines."""

import argparse
from dataclasses import dataclass
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
for path in (ROOT, ROOT / "Experiments"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from pelinn.model import PELiNNQEM, physics_loss  # noqa: E402
try:  # Optional: only needed when synthesizing Qiskit datasets.
    from pelinn.data.qiskit_dataset import Sample, synthesize_samples  # type: ignore
except Exception:  # pragma: no cover - fallback for environments without Qiskit
    synthesize_samples = None

    @dataclass
    class Sample:  # type: ignore
        x: np.ndarray
        y_noisy: float
        y_ideal: float
        meta: Dict


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/demo_dataset.npz",
        help="Path to NPZ dataset containing arrays X (features), Y (targets), metadata. Ignored when --save-samples is set.",
    )
    parser.add_argument(
        "--save-samples",
        type=str,
        default=None,
        help="If set, synthesize a Qiskit dataset and save to this NPZ path before training.",
    )
    parser.add_argument("--num-qubits", type=int, default=4, help="Number of qubits per synthesized circuit.")
    parser.add_argument("--num-circuits", type=int, default=32, help="Number of base circuits to synthesize.")
    parser.add_argument("--shots-noisy", type=int, default=4096, help="Shot count for noisy estimation.")
    parser.add_argument("--shots-ideal", type=int, default=0, help="Shot count for ideal estimation (0 => exact).")
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
    parser.add_argument(
        "--baselines",
        type=str,
        default="",
        help="Comma-separated baselines to evaluate on the validation split (e.g., zne,cdr).",
    )
    parser.add_argument(
        "--zne-scale-factors",
        type=float,
        nargs="+",
        default=(1.0, 2.0, 3.0),
        help="Noise scale factors for ZNE.",
    )
    parser.add_argument(
        "--cdr-training-circuits",
        type=int,
        default=30,
        help="Training circuits for CDR.",
    )
    parser.add_argument(
        "--baseline-max-samples",
        type=int,
        default=0,
        help="Limit baseline evaluation to first N validation samples (0 => all).",
    )
    return parser.parse_args(argv)


@dataclass
class BaselineConfig:
    zne_scale_factors: Sequence[float]
    cdr_training_circuits: int


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


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def make_circuit_family(num_qubits: int, num_circuits: int, rng: random.Random):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp

    circuits: List[QuantumCircuit] = []
    observables: List[SparsePauliOp] = []
    for _ in range(num_circuits):
        qc = QuantumCircuit(num_qubits)
        for qubit in range(num_qubits):
            qc.h(qubit)
        for _layer in range(4):
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
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=X, Y=Y, metadata=metadata)


def synthesize_and_save_dataset(args: argparse.Namespace, rng: random.Random) -> Path:
    if synthesize_samples is None:
        raise RuntimeError("Qiskit/Aer is required to synthesize datasets (missing qiskit or qiskit-aer).")
    circuits, observables = make_circuit_family(args.num_qubits, args.num_circuits, rng)
    samples = synthesize_samples(
        circuits,
        observables,
        default_noise_grid(),
        shots_noisy=args.shots_noisy,
        shots_ideal=args.shots_ideal,
    )
    dataset_path = resolve_path(args.save_samples)
    save_samples_npz(samples, dataset_path)
    logging.info("Saved synthesized samples to %s", dataset_path)
    return dataset_path


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


def _unwrap_metadata(metadata: object) -> object:
    if isinstance(metadata, np.ndarray) and metadata.dtype == object and metadata.ndim == 0:
        try:
            return metadata.item()
        except Exception:
            return metadata
    return metadata


def _extract_metadata_entry(entry: object, idx: int) -> Dict:
    if isinstance(entry, dict):
        meta = dict(entry)
    elif isinstance(entry, (int, np.integer)):
        meta = {"circuit_index": int(entry)}
    else:
        meta = {"value": entry}
    meta.setdefault("circuit_index", meta.get("circuit_index", idx))
    meta.setdefault("sample_index", idx)
    return meta


def load_npz_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict] | None]:
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
    metadata = _unwrap_metadata(blob.get("metadata"))
    circuit_ids = np.arange(len(X))
    meta_entries = None
    if metadata is not None:
        try:
            if isinstance(metadata, dict) and "circuit_index" in metadata:
                circuit_ids = np.array(metadata["circuit_index"], dtype=int)
            elif isinstance(metadata, (list, tuple, np.ndarray)):
                meta_entries = []
                for idx in range(len(X)):
                    entry = metadata[idx] if idx < len(metadata) else {}
                    meta_entries.append(_extract_metadata_entry(entry, idx))
                circuit_ids = np.array(
                    [(_extract_circuit_idx(entry, idx)) for idx, entry in enumerate(meta_entries)],
                    dtype=int,
                )
        except Exception as exc:
            logging.warning("Failed to parse circuit indices from metadata (%s); using sequential ids.", exc)
            meta_entries = None
    logging.debug("Loaded dataset: X%s, Y%s", X.shape, Y.shape)
    return X, Y, circuit_ids, meta_entries


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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if y_true.size == 0 or y_pred.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "max_abs": float("nan")}
    diff = y_pred - y_true
    mse = float(np.mean(diff ** 2))
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(mse)),
        "max_abs": float(np.max(np.abs(diff))),
    }


def run_benchmarks(
    baseline_names: Sequence[str],
    meta_entries: Sequence[Dict] | None,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cfg: BaselineConfig,
    max_samples: int,
) -> Dict[str, object] | None:
    if not baseline_names:
        return None
    if not meta_entries:
        logging.warning("Baseline evaluation requested, but dataset metadata is missing.")
        return None
    try:
        import warnings
        from qiskit import QuantumCircuit, transpile
        from qiskit.quantum_info import SparsePauliOp
        from qiskit_aer import AerSimulator
        from pelinn.data.qiskit_dataset import make_noise_model, _expectation_from_counts

        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message="The class ``qiskit.primitives.backend_estimator.BackendEstimator`` is deprecated",
        )
        try:
            from qiskit.primitives import BackendEstimator
        except Exception:
            try:
                from qiskit.primitives import Estimator as BackendEstimator  # type: ignore
            except Exception:
                BackendEstimator = None
    except Exception as exc:
        logging.warning("Skipping baselines: Qiskit/Aer unavailable (%s).", exc)
        return None

    def resolve_baseline(name: str):
        lname = name.lower()
        if lname == "zne":
            from baselines.zne_mitiq import mitigate_with_zne

            return mitigate_with_zne
        if lname == "cdr":
            from baselines.cdr_mitiq import mitigate_with_cdr

            return mitigate_with_cdr
        raise ValueError(f"Unknown baseline '{name}'.")

    class NoisyExecutorPool:
        def __init__(self):
            self._cache: Dict[tuple, tuple[AerSimulator, object | None]] = {}

        def get_executor(self, noise_cfg: Dict, observable, shots: int | None):
            key = tuple(sorted(noise_cfg.items()))
            if key not in self._cache:
                nm = make_noise_model(**noise_cfg)
                backend = AerSimulator(noise_model=nm)
                estimator = BackendEstimator(backend) if BackendEstimator is not None else None
                self._cache[key] = (backend, estimator)
            backend, estimator = self._cache[key]
            eff_shots = None if shots is None or shots <= 0 else int(shots)

            def _executor(qc) -> float:
                compiled = transpile(qc, backend)
                if estimator is not None:
                    result = estimator.run(
                        circuits=compiled,
                        observables=observable,
                        shots=eff_shots,
                    ).result()
                    try:
                        return float(result.values[0])
                    except Exception:
                        return float(result[0])
                job = backend.run(compiled, shots=eff_shots)
                res = job.result()
                try:
                    counts = res.get_counts()
                    shots_used = eff_shots if eff_shots is not None and eff_shots > 0 else sum(counts.values())
                    return float(_expectation_from_counts(observable, counts, shots_used))
                except Exception:
                    try:
                        from qiskit.quantum_info import Statevector

                        sv = Statevector.from_instruction(compiled)
                        return float(sv.expectation_value(observable).real)
                    except Exception as exc:
                        raise RuntimeError("Unable to execute observable: " + str(exc))

            _executor.__annotations__["return"] = float
            return _executor

    valid_pairs: List[Tuple[int, Dict]] = []
    dropped = 0
    for idx, meta in enumerate(meta_entries):
        if not isinstance(meta, dict):
            dropped += 1
            continue
        qc = meta.get("qc")
        if qc is None and "circuit_qasm" in meta:
            try:
                qc = QuantumCircuit.from_qasm_str(meta["circuit_qasm"])
            except Exception:
                qc = None
        obs = meta.get("observable")
        noise = meta.get("noise")
        if qc is None or obs is None or noise is None:
            dropped += 1
            continue
        if not isinstance(qc, QuantumCircuit) or not isinstance(obs, SparsePauliOp) or not isinstance(noise, dict):
            dropped += 1
            continue
        resolved = dict(meta)
        resolved["qc"] = qc
        resolved["observable"] = obs
        resolved["noise"] = noise
        valid_pairs.append((idx, resolved))

    if not valid_pairs:
        logging.warning("Baseline evaluation requested, but metadata lacks qc/noise/observable entries.")
        return None
    if dropped:
        logging.warning(
            "Baseline evaluation dropping %d/%d validation samples without usable metadata.",
            dropped,
            len(meta_entries),
        )
    if max_samples > 0 and len(valid_pairs) > max_samples:
        valid_pairs = valid_pairs[:max_samples]
        logging.info("Restricting baseline evaluation to first %d samples.", len(valid_pairs))

    indices = [idx for idx, _ in valid_pairs]
    meta_subset = [meta for _, meta in valid_pairs]
    y_true_subset = y_true[indices]
    y_pred_subset = y_pred[indices]

    results: Dict[str, Dict[str, float]] = {}
    results["PE-LINN"] = compute_metrics(y_true_subset, y_pred_subset)
    y_noisy = None
    try:
        y_noisy = np.array([float(meta["y_noisy"]) for meta in meta_subset], dtype=np.float64)
        if np.isnan(y_noisy).any():
            y_noisy = None
    except Exception:
        y_noisy = None
    if y_noisy is not None:
        results["NOISY"] = compute_metrics(y_true_subset, y_noisy)

    pool = NoisyExecutorPool()
    for name in baseline_names:
        try:
            baseline_fn = resolve_baseline(name)
        except RuntimeError as exc:
            logging.warning("Skipping %s baseline: %s", name.upper(), exc)
            continue
        except ValueError as exc:
            logging.warning(str(exc))
            continue

        preds: List[float] = []
        for idx, sample in enumerate(meta_subset, 1):
            executor = pool.get_executor(
                sample["noise"],
                sample["observable"],
                sample.get("shots_noisy"),
            )
            if name.lower() == "zne":
                value = baseline_fn(
                    executor=executor,
                    circuit=sample["qc"],
                    observable=sample["observable"],
                    scale_factors=tuple(cfg.zne_scale_factors),
                )
            elif name.lower() == "cdr":
                value = baseline_fn(
                    executor=executor,
                    circuit=sample["qc"],
                    observable=sample["observable"],
                    num_training_circuits=cfg.cdr_training_circuits,
                )
            else:
                raise AssertionError("Unexpected baseline dispatch.")
            preds.append(float(value))
            if idx % 10 == 0:
                logging.debug("%s baseline progress: %d/%d", name.upper(), idx, len(meta_subset))
        results[name.upper()] = compute_metrics(y_true_subset, np.array(preds, dtype=np.float64))

    print("\n=== Benchmark metrics (lower is better) ===")
    for name, metric in results.items():
        print(
            f"{name:>8} | MAE: {metric['mae']:.5f} | RMSE: {metric['rmse']:.5f} | MAX|err|: {metric['max_abs']:.5f}"
        )
    return {"num_samples": int(len(y_true_subset)), "metrics": results}


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
    benchmarks: Dict[str, object] | None = None,
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
    if benchmarks is not None:
        summary["benchmarks"] = benchmarks
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    logging.info("Wrote training summary to %s", summary_path)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    logging.getLogger("qiskit").setLevel(logging.WARNING)
    logging.getLogger("qiskit.transpiler").setLevel(logging.WARNING)
    logging.getLogger("qiskit.transpiler.passes").setLevel(logging.WARNING)
    logging.getLogger("qiskit.transpiler.run").setLevel(logging.WARNING)
    set_seed(args.seed)
    rng = random.Random(args.seed)

    dataset_path = None
    if args.save_samples:
        dataset_path = synthesize_and_save_dataset(args, rng)
    else:
        dataset_path = resolve_path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    logging.info("Loading NPZ dataset from %s", dataset_path)

    X, Y, circuit_ids, meta_entries = load_npz_dataset(dataset_path)
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
    val_meta_entries = [meta_entries[idx] for idx in val_idx] if meta_entries is not None and val_idx.size else None

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

    benchmark_summary = None
    if val_loader:
        metrics = evaluate(model, val_loader, device)
        final_metrics.update({"mae": metrics["mae"], "rmse": metrics["rmse"]})
        y_true = metrics["y_true"]
        y_pred = metrics["y_pred"]
        baseline_names = [name.strip() for name in args.baselines.split(",") if name.strip()]
        if baseline_names:
            benchmark_summary = run_benchmarks(
                baseline_names=baseline_names,
                meta_entries=val_meta_entries,
                y_true=y_true,
                y_pred=y_pred,
                cfg=BaselineConfig(
                    zne_scale_factors=args.zne_scale_factors,
                    cdr_training_circuits=args.cdr_training_circuits,
                ),
                max_samples=args.baseline_max_samples,
            )

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
    write_summary(summary_path, model, final_metrics, args, dataset_info, benchmarks=benchmark_summary)


if __name__ == "__main__":
    main()

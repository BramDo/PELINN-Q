#!/usr/bin/env python3
"""Evaluate PE-LiNN QEM against standard Mitiq baselines on synthetic datasets."""

from __future__ import annotations

import argparse
import logging
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _ensure_project_root() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = _ensure_project_root()

# Suppress noisy DeprecationWarnings from legacy BackendEstimator until we migrate to V2.
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="The class ``qiskit.primitives.backend_estimator.BackendEstimator`` is deprecated",
)

from pelinn.model import PELiNNQEM  # noqa: E402
from pelinn.data.qiskit_dataset import make_noise_model, synthesize_samples  # noqa: E402
from qiskit import QuantumCircuit, transpile  # noqa: E402
from qiskit.quantum_info import SparsePauliOp  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402
from qiskit.primitives import BackendEstimator  # noqa: E402


@dataclass
class BaselineConfig:
    zne_scale_factors: Sequence[float]
    cdr_training_circuits: int


def compute_feature_stats(samples) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-feature mean and std for normalisation."""
    features = np.stack([s.x for s in samples]).astype(np.float32)
    mean = torch.from_numpy(features.mean(axis=0))
    std = torch.from_numpy(features.std(axis=0)).clamp_min(1e-6)
    return mean, std


class SamplesDataset(Dataset):
    def __init__(self, samples, mean: torch.Tensor | None = None, std: torch.Tensor | None = None):
        self.X = torch.tensor(np.stack([s.x for s in samples]), dtype=torch.float32)
        if mean is not None and std is not None:
            mean_t = mean.to(self.X)
            std_t = std.to(self.X)
            if mean_t.ndim == 1:
                mean_t = mean_t.unsqueeze(0)
            if std_t.ndim == 1:
                std_t = std_t.unsqueeze(0)
            self.X = (self.X - mean_t) / std_t
        self.y = torch.tensor([s.y_ideal for s in samples], dtype=torch.float32)
        self.cid = torch.tensor(
            [int(s.meta.get("circuit_index", id(s.meta["qc"]))) for s in samples],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.cid[idx]


class NoisyExecutorPool:
    """Cache noisy backends to avoid rebuilding Aer simulators per call."""

    def __init__(self):
        self._cache: Dict[tuple, tuple[AerSimulator, BackendEstimator]] = {}

    def get_executor(self, noise_cfg: Dict, observable: SparsePauliOp, shots: int | None):
        key = tuple(sorted(noise_cfg.items()))
        if key not in self._cache:
            nm = make_noise_model(**noise_cfg)
            backend = AerSimulator(noise_model=nm)
            estimator = BackendEstimator(backend)
            self._cache[key] = (backend, estimator)
        backend, estimator = self._cache[key]
        eff_shots = None if shots is None or shots <= 0 else int(shots)

        def _executor(qc) -> float:
            compiled = transpile(qc, backend)
            result = estimator.run(
                circuits=compiled,
                observables=observable,
                shots=eff_shots,
            ).result()
            return float(result.values[0])

        _executor.__annotations__["return"] = float
        return _executor


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-qubits", type=int, default=4, help="Number of qubits per circuit.")
    parser.add_argument("--num-circuits", type=int, default=24, help="Number of base circuits to generate.")
    parser.add_argument("--shots-noisy", type=int, default=4096, help="Shot count for noisy estimations.")
    parser.add_argument("--shots-ideal", type=int, default=0, help="Shot count for ideal (0 => exact).")
    parser.add_argument("--test-fraction", type=float, default=0.25, help="Fraction of circuits reserved for evaluation.")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs for PE-LiNN.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay.")
    parser.add_argument("--alpha-inv", type=float, default=0.1, help="Invariance penalty weight.")
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        choices=("mse", "huber"),
        help="Primary regression loss for PE-LiNN (mse or huber/smooth L1).",
    )
    parser.add_argument(
        "--huber-beta",
        type=float,
        default=0.1,
        help="Beta parameter for Huber (smooth L1) loss when --loss huber.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Global RNG seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Preferred device (falls back to cpu).")
    parser.add_argument("--baselines", type=str, default="zne,cdr", help="Comma-separated baselines to evaluate.")
    parser.add_argument("--hid-dim", type=int, default=96, help="Hidden dimension of the PE-LiNN model.")
    parser.add_argument("--steps", type=int, default=6, help="Number of recurrent integration steps.")
    parser.add_argument("--dt", type=float, default=0.25, help="Integration step size for the liquid cell.")
    parser.add_argument(
        "--tanh-head",
        action="store_true",
        help="Use a tanh head to bound outputs to [-1, 1] after training.",
    )
    parser.add_argument("--zne-scale-factors", type=float, nargs="+", default=(1.0, 2.0, 3.0), help="Noise scale factors for ZNE.")
    parser.add_argument("--cdr-training-circuits", type=int, default=30, help="Training circuits for CDR.")
    parser.add_argument("--max-eval-samples", type=int, default=32, help="Limit evaluation to first N samples (0 => all).")
    parser.add_argument(
        "--preview-samples",
        type=int,
        default=0,
        help="If >0, print a short table of raw predictions for the first N evaluation samples.",
    )
    parser.add_argument(
        "--clamp-outputs",
        type=float,
        default=None,
        help="If set, clamp PE-LiNN predictions to [-value, value] before computing metrics.",
    )
    parser.add_argument(
        "--outlier-z",
        type=float,
        default=None,
        help=(
            "If set, drop evaluation samples whose |prediction-ideal| exceeds mean(abs err) + z * std(abs err). "
            "Applies after optional clamping."
        ),
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_circuit_family(num_qubits: int, num_circuits: int, rng: random.Random) -> tuple[List[QuantumCircuit], List[SparsePauliOp]]:
    circuits: List[QuantumCircuit] = []
    observables: List[SparsePauliOp] = []
    for _ in range(num_circuits):
        qc = QuantumCircuit(num_qubits)
        for q in range(num_qubits):
            qc.h(q)
        for q in range(0, num_qubits - 1):
            qc.cx(q, q + 1)
        for layer in range(2):
            for q in range(num_qubits):
                qc.rz(rng.uniform(-np.pi, np.pi), q)
            for q in range(0, num_qubits - 1):
                qc.cx(q, q + 1)
            qc.rx(rng.uniform(-np.pi, np.pi), rng.randrange(num_qubits))
        qc.measure_all(False)
        circuits.append(qc)
        observables.append(SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1.0)]))
    return circuits, observables


def default_noise_grid() -> List[Dict[str, float]]:
    return [
        {"p1_depol": 0.001, "p2_depol": 0.01, "p_amp": 0.001, "readout_p01": 0.02, "readout_p10": 0.02},
        {"p1_depol": 0.003, "p2_depol": 0.02, "p_amp": 0.002, "readout_p01": 0.03, "readout_p10": 0.03},
    ]


def split_samples(samples, test_fraction: float, seed: int):
    by_circuit: Dict[int, List] = {}
    for sample in samples:
        key = int(sample.meta.get("circuit_index", id(sample.meta["qc"])))
        by_circuit.setdefault(key, []).append(sample)
    circuit_ids = list(by_circuit.keys())
    rng = random.Random(seed)
    rng.shuffle(circuit_ids)

    if test_fraction <= 0.0:
        n_test = 0
    elif test_fraction >= 1.0:
        n_test = len(circuit_ids)
    else:
        n_test = max(1, int(round(len(circuit_ids) * test_fraction)))

    test_ids = set(circuit_ids[:n_test])
    train, test = [], []
    for cid, group in by_circuit.items():
        (test if cid in test_ids else train).extend(group)
    rng.shuffle(train)
    rng.shuffle(test)
    if not train:
        raise ValueError("Train split is empty; reduce --test-fraction.")
    if not test:
        raise ValueError("Test split is empty; increase --test-fraction.")
    return train, test


def make_groups(cids: torch.Tensor) -> List[List[int]]:
    groups: Dict[int, List[int]] = {}
    for idx, cid in enumerate(cids.cpu().tolist()):
        groups.setdefault(int(cid), []).append(idx)
    return list(groups.values())


def train_model(
    model: PELiNNQEM,
    dataset: SamplesDataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    alpha_inv: float,
    loss_type: str,
    huber_beta: float,
) -> None:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        running = 0.0
        for X, y, cid in loader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss = model.compute_loss(
                pred,
                y,
                groups=make_groups(cid),
                alpha_inv=alpha_inv,
                loss_type=loss_type,
                huber_beta=huber_beta,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
        if epoch == 0 or (epoch + 1) % max(1, epochs // 5) == 0:
            logging.info("Epoch %d/%d | loss=%.5f", epoch + 1, epochs, running / max(1, len(loader)))


def evaluate_model(
    model: PELiNNQEM,
    samples,
    device: torch.device,
    batch_size: int,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> np.ndarray:
    model.eval()
    dataset = SamplesDataset(samples, mean=mean, std=std)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds: List[torch.Tensor] = []
    with torch.no_grad():
        for X, _, _ in loader:
            preds.append(model(X.to(device)).cpu())
    return torch.cat(preds).numpy()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    diff = y_pred - y_true
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    return {"mae": mae, "rmse": float(np.sqrt(mse)), "max_abs": max_abs}


def resolve_baseline(name: str):
    lname = name.lower()
    if lname == "zne":
        from baselines.zne_mitiq import mitigate_with_zne

        return mitigate_with_zne
    if lname == "cdr":
        from baselines.cdr_mitiq import mitigate_with_cdr

        return mitigate_with_cdr
    if lname == "pec":
        from baselines.pec_mitiq import mitigate_with_pec

        return mitigate_with_pec
    raise ValueError(f"Unknown baseline '{name}'.")


def evaluate_baselines(
    baseline_names: Iterable[str],
    samples,
    executor_pool: NoisyExecutorPool,
    cfg: BaselineConfig,
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    y_true = np.array([s.y_ideal for s in samples], dtype=np.float64)
    for name in baseline_names:
        try:
            baseline_fn = resolve_baseline(name)
        except RuntimeError as exc:
            logging.warning("Skipping %s baseline: %s", name.upper(), exc)
            continue
        except ValueError as exc:
            logging.warning(str(exc))
            continue

        if name.lower() == "pec":
            logging.warning("Skipping PEC baseline: provide a calibrated representation to enable it.")
            continue

        preds: List[float] = []
        for idx, sample in enumerate(samples, 1):
            executor = executor_pool.get_executor(
                sample.meta["noise"],
                sample.meta["observable"],
                sample.meta.get("shots_noisy"),
            )
            if name.lower() == "zne":
                value = baseline_fn(
                    executor=executor,
                    circuit=sample.meta["qc"],
                    observable=sample.meta["observable"],
                    scale_factors=tuple(cfg.zne_scale_factors),
                )
            elif name.lower() == "cdr":
                value = baseline_fn(
                    executor=executor,
                    circuit=sample.meta["qc"],
                    observable=sample.meta["observable"],
                    num_training_circuits=cfg.cdr_training_circuits,
                )
            else:
                raise AssertionError("Unexpected baseline dispatch.")
            preds.append(float(value))
            if idx % 10 == 0:
                logging.debug("%s baseline progress: %d/%d", name.upper(), idx, len(samples))
        metrics[name.upper()] = compute_metrics(y_true, np.array(preds, dtype=np.float64))
    return metrics


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    # Silence verbose Qiskit transpiler diagnostics; escalate severity if needed.
    logging.getLogger("qiskit").setLevel(logging.WARNING)
    logging.getLogger("qiskit.transpiler").setLevel(logging.WARNING)
    logging.getLogger("qiskit.transpiler.run").setLevel(logging.WARNING)
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    logging.info("Using device: %s", device)

    rng = random.Random(args.seed)
    circuits, observables = make_circuit_family(args.num_qubits, args.num_circuits, rng)
    noise_grid = default_noise_grid()

    samples = synthesize_samples(
        circuits,
        observables,
        noise_grid,
        shots_noisy=args.shots_noisy,
        shots_ideal=args.shots_ideal,
    )
    train_samples, test_samples = split_samples(samples, args.test_fraction, args.seed)

    model = PELiNNQEM(
        in_dim=train_samples[0].x.shape[0],
        hid_dim=args.hid_dim,
        steps=args.steps,
        dt=args.dt,
        use_tanh_head=args.tanh_head,
    )
    model.to(device)
    feature_mean, feature_std = compute_feature_stats(train_samples)
    logging.info("Normalising features using training mean/std.")
    train_dataset = SamplesDataset(train_samples, mean=feature_mean, std=feature_std)
    train_model(
        model=model,
        dataset=train_dataset,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        alpha_inv=args.alpha_inv,
        loss_type=args.loss,
        huber_beta=args.huber_beta,
    )

    eval_samples = list(test_samples)
    if args.max_eval_samples and args.max_eval_samples > 0:
        eval_samples = eval_samples[: args.max_eval_samples]
        logging.info("Restricting evaluation to first %d samples.", len(eval_samples))

    rng.shuffle(eval_samples)
    y_true = np.array([s.y_ideal for s in eval_samples], dtype=np.float64)
    y_noisy = np.array([s.y_noisy for s in eval_samples], dtype=np.float64)
    circuit_ids = [int(s.meta.get("circuit_index", id(s.meta["qc"]))) for s in eval_samples]
    pelinn_preds = evaluate_model(
        model,
        eval_samples,
        device,
        args.batch_size,
        mean=feature_mean,
        std=feature_std,
    )
    if args.clamp_outputs is not None and args.clamp_outputs > 0:
        pelinn_preds = np.clip(pelinn_preds, -args.clamp_outputs, args.clamp_outputs)
    kept_indices = np.arange(len(eval_samples))
    if args.outlier_z is not None and args.outlier_z > 0 and len(eval_samples) > 0:
        abs_err = np.abs(pelinn_preds - y_true)
        mean_err = float(abs_err.mean())
        std_err = float(abs_err.std())
        threshold = mean_err + args.outlier_z * std_err
        mask = abs_err <= threshold if std_err > 0 else abs_err <= mean_err
        kept = np.nonzero(mask)[0]
        if kept.size == 0:
            logging.warning("Outlier filter would drop all samples; keeping original set.")
        else:
            dropped = len(eval_samples) - kept.size
            if dropped > 0:
                logging.info(
                    "Dropped %d/%d evaluation samples with |err| > %.5f (mean=%.5f, std=%.5f).",
                    dropped,
                    len(eval_samples),
                    threshold,
                    mean_err,
                    std_err,
                )
                eval_samples = [eval_samples[i] for i in kept]
                y_true = y_true[kept]
                y_noisy = y_noisy[kept]
                pelinn_preds = pelinn_preds[kept]
                circuit_ids = [circuit_ids[i] for i in kept]
                kept_indices = kept

    results = {
        "NOISY": compute_metrics(y_true, y_noisy),
        "PE-LINN": compute_metrics(y_true, pelinn_preds),
    }

    baseline_names = [name.strip() for name in args.baselines.split(",") if name.strip()]
    if baseline_names:
        pool = NoisyExecutorPool()
        baseline_cfg = BaselineConfig(
            zne_scale_factors=args.zne_scale_factors,
            cdr_training_circuits=args.cdr_training_circuits,
        )
        results.update(evaluate_baselines(baseline_names, eval_samples, pool, baseline_cfg))

    logging.info("Evaluation completed on %d samples (%d circuits).", len(eval_samples), len({s.meta['circuit_index'] for s in eval_samples}))

    if args.preview_samples and args.preview_samples > 0:
        preview = min(args.preview_samples, len(eval_samples))
        print("\nFirst few predictions (circuit id, noisy, PE-LiNN, ideal, abs err):")
        for idx in range(preview):
            err = abs(pelinn_preds[idx] - y_true[idx])
            orig_idx = int(kept_indices[idx]) if isinstance(kept_indices, np.ndarray) else idx
            print(
                f"  #{orig_idx:02d} cid={circuit_ids[idx]} "
                f"noisy={y_noisy[idx]: .5f} pelinn={pelinn_preds[idx]: .5f} "
                f"ideal={y_true[idx]: .5f} |err|={err: .5f}"
            )

    print("\n=== Regression metrics (lower is better) ===")
    for name, metric in results.items():
        print(
            f"{name:>8} | MAE: {metric['mae']:.5f} | RMSE: {metric['rmse']:.5f} | MAX|err|: {metric['max_abs']:.5f}"
        )


if __name__ == "__main__":
    main()

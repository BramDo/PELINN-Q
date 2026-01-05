#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "Experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from qiskit.quantum_info import SparsePauliOp  # noqa: E402

from scripts.eval_benchmarks import (  # noqa: E402
    NoisyExecutorPool,
    default_noise_grid,
    make_circuit_family,
    split_samples,
)
from pelinn.data.qiskit_dataset import synthesize_samples  # noqa: E402

try:
    # Mitiq folding, already used by your repo baseline :contentReference[oaicite:3]{index=3}
    from mitiq.zne.scaling import folding
except Exception as exc:
    raise RuntimeError("This script requires mitiq (e.g., mitiq==0.48.1).") from exc

from diagnostics.zne_diagnostics import zne_with_ci, detect_zne_blowups  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-qubits", type=int, default=4)
    ap.add_argument("--num-circuits", type=int, default=24)
    ap.add_argument("--shots-noisy", type=int, default=4096)
    ap.add_argument("--shots-ideal", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--zne-scale-factors", type=float, nargs="+", default=(1.0, 2.0, 3.0))
    ap.add_argument("--zne-repeats", type=int, default=25)
    ap.add_argument("--zne-degree", type=int, default=1)
    ap.add_argument("--ci-level", type=float, default=0.95)
    ap.add_argument("--max-eval-samples", type=int, default=0)
    ap.add_argument("--err-ratio-thresh", type=float, default=2.0)
    ap.add_argument("--abs-err-thresh", type=float, default=0.25)
    ap.add_argument("--ci-width-thresh", type=float, default=0.25)
    return ap.parse_args()


def main():
    args = parse_args()

    rng = np.random.default_rng(args.seed)
    circuits, observables = make_circuit_family(args.num_qubits, args.num_circuits, rng=rng)

    noise_grid = default_noise_grid()
    samples = synthesize_samples(
        circuits=circuits,
        observables=observables,
        noise_grid=noise_grid,
        shots_noisy=args.shots_noisy,
        shots_ideal=args.shots_ideal,
    )

    # Reuse the same train/test grouping logic as eval_benchmarks.py :contentReference[oaicite:4]{index=4}
    _train, test = split_samples(samples, test_fraction=0.25, seed=args.seed)

    if args.max_eval_samples and args.max_eval_samples > 0:
        test = test[: int(args.max_eval_samples)]

    pool = NoisyExecutorPool()

    cids = []
    qcs = []
    ideals = []
    noisies = []
    znes = []

    for s in test:
        executor = pool.get_executor(s.meta["noise"], s.meta["observable"], s.meta.get("shots_noisy"))
        # Noisy point estimate at scale=1
        y_noisy = float(executor(s.meta["qc"]))

        # Ideal from dataset
        y_ideal = float(s.y_ideal)

        # CI-aware ZNE
        zr = zne_with_ci(
            executor=executor,
            circuit=s.meta["qc"],
            scale_noise=folding.fold_global,
            scale_factors=tuple(args.zne_scale_factors),
            n_repeats=args.zne_repeats,
            degree=args.zne_degree,
            ci_level=args.ci_level,
            rng_seed=args.seed,
        )

        cids.append(int(s.meta.get("circuit_index", id(s.meta["qc"]))))
        qcs.append(s.meta["qc"])
        ideals.append(y_ideal)
        noisies.append(y_noisy)
        znes.append(zr)

    blowups = detect_zne_blowups(
        cids, qcs, ideals, noisies, znes,
        err_ratio_thresh=args.err_ratio_thresh,
        abs_err_thresh=args.abs_err_thresh,
        ci_width_thresh=args.ci_width_thresh,
    )

    print("=== CI-aware ZNE diagnostics ===")
    print(f"eval circuits: {len(cids)}")
    print(f"blow-ups: {len(blowups)}")
    print("")
    if blowups:
        print("Top blow-ups (worst first):")
        for b in blowups[: min(20, len(blowups))]:
            print(
                f"cid={b.circuit_id} depth={b.depth} cx={b.n_cx}  "
                f"|noisy_err|={b.noisy_err:.5f} |zne_err|={b.zne_err:.5f} "
                f"CI_width={b.zne_ci_width:.5f} reason={b.reason}"
            )

    # Optional: summarize variance inflation
    ci_widths = np.array([zr.ci_high - zr.ci_low for zr in znes], dtype=np.float64)
    print("")
    print("CI width summary:")
    print(f"mean={ci_widths.mean():.5f}  median={np.median(ci_widths):.5f}  p90={np.quantile(ci_widths, 0.90):.5f}  max={ci_widths.max():.5f}")


if __name__ == "__main__":
    main()

# Classes in `scripts/eval_benchmarks.py`

## `BaselineConfig`
A `@dataclass` that groups hyperparameters for baseline methods.

### Fields
- **`zne_scale_factors`** (`Sequence[float]`): scaling factors for ZNE folding.
- **`cdr_training_circuits`** (`int`): number of training circuits for CDR.

## `SamplesDataset`
A PyTorch `Dataset` for normalised sample collections used during training and
evaluation phases.

### Key attributes
- **`X`**: `torch.Tensor` containing (optionally) normalised feature vectors.
- **`y`**: `torch.Tensor` with ideal expectation values.
- **`cid`**: `torch.Tensor` with circuit identifiers to support invariance
  losses.

### Core methods
- **`__len__`**: returns the number of samples.
- **`__getitem__(index)`**: yields `(features, target, circuit_id)` for the
  provided index.

## `NoisyExecutorPool`
Maintains a cache of Qiskit `AerSimulator` backends and associated
`BackendEstimator` instances to speed up repeated evaluations.

### Key attributes
- **`_cache`**: dictionary mapping noise configurations (as sorted tuples) to
  `(AerSimulator, BackendEstimator)` pairs.

### Core methods
- **`get_executor(noise_cfg, observable, shots)`**: returns an executor function
  that transpiles a circuit for the specified noise configuration and then
  evaluates the expectation value of the provided observable. The helper manages
  backend caching and ensures the executor honours shot counts.

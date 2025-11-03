# Classes in `scripts/train_pelinn.py`

## `QemDataset`
A `torch.utils.data.Dataset` implementation that stores synthesised quantum
error mitigation samples.

### Key attributes
- **`X`**: `numpy.ndarray` containing the composite feature vector per sample.
- **`y`**: `numpy.ndarray` with ideal expectation values per sample.
- **`cid`**: `numpy.ndarray` with circuit identifiers used to group
  measurements.

### Core methods
- **`__len__`**: returns the number of samples in the dataset.
- **`__getitem__(index)`**: returns `(features, target, circuit_id)` for
  index-based access, suitable for use with a PyTorch `DataLoader`.

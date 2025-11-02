# Klassen in `scripts/train_pelinn.py`

## `QemDataset`
Een `torch.utils.data.Dataset`-implementatie die gesynthetiseerde
quantum-error-mitigatie-samples bevat.

### Belangrijkste attributen
- **`X`**: `numpy.ndarray` met de samengestelde featurevector per sample.
- **`y`**: `numpy.ndarray` met ideale verwachtingswaarden per sample.
- **`cid`**: `numpy.ndarray` met circuit-identificaties voor het groeperen van
  metingen.

### Kernmethoden
- **`__len__`**: retourneert het aantal samples in de dataset.
- **`__getitem__(index)`**: geeft een tuple `(features, target, circuit_id)`
  voor index-gebaseerde toegang, geschikt voor gebruik met PyTorch
  `DataLoader`.

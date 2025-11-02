# Klassen in `scripts/eval_benchmarks.py`

## `BaselineConfig`
Een `@dataclass` die hyperparameters voor baseline-methoden bundelt.

### Velden
- **`zne_scale_factors`** (`Sequence[float]`): schaalfactoren voor ZNE-folding.
- **`cdr_training_circuits`** (`int`): aantal trainingscircuits voor CDR.

## `SamplesDataset`
Een PyTorch `Dataset` voor genormaliseerde samplecollecties die tijdens
trainings- en evaluatiefases worden gebruikt.

### Belangrijkste attributen
- **`X`**: `torch.Tensor` met (optioneel) genormaliseerde featurevectoren.
- **`y`**: `torch.Tensor` met ideale verwachtingswaarden.
- **`cid`**: `torch.Tensor` met circuit-identificaties om invariantieverliezen te
  ondersteunen.

### Kernmethoden
- **`__len__`**: retourneert het aantal samples.
- **`__getitem__(index)`**: levert `(features, target, circuit_id)` voor de
  gegeven index.

## `NoisyExecutorPool`
Beheert een cache van Qiskit `AerSimulator`-backends en bijbehorende
`BackendEstimator`-instanties om herhaalde evaluaties te versnellen.

### Belangrijkste attributen
- **`_cache`**: dictionary die ruisconfiguraties (als gesorteerde tuples) mapt
  naar `(AerSimulator, BackendEstimator)`-paren.

### Kernmethoden
- **`get_executor(noise_cfg, observable, shots)`**: geeft een uitvoerfunctie die
  een circuit transpileert voor de bijhorende ruisconfiguratie en daarna de
  verwachting van het opgegeven observabelen berekent. De helper zorgt voor
  caching van backends en stelt de uitvoerfunctie in staat om shot-aantallen te
  honoreren.

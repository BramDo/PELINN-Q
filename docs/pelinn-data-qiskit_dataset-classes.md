# Klassen in `pelinn/data/qiskit_dataset.py`

## `Sample`
Een `@dataclass` die één datapunt voor quantum error mitigation voorstelt.

### Velden
- **`x`** (`numpy.ndarray`): samengestelde featurevector met circuitstatistieken,
  ruisparameters en meetresultaten.
- **`y_noisy`** (`float`): gemeten verwachtingswaarde onder een noisy backend.
- **`y_ideal`** (`float`): referentie-verwachtingswaarde uit een ideale simulatie.
- **`meta`** (`Dict`): aanvullende metadata, zoals het oorspronkelijke circuit,
  de gebruikte ruisconfiguratie en shot-aantallen.

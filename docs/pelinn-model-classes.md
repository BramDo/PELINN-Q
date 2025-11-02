# Klassen in `pelinn/model.py`

## `LTCCell`
Een `torch.nn.Module` die een Liquid Time-Constant (LTC) cel implementeert met
regularisatietermen voor gating-activaties en parameter `A`.

### Belangrijkste attributen
- **`W_tx`, `W_th`, `b_t`**: lineaire lagen en bias voor de tijdconstante
  dynamiek.
- **`W_gx`, `W_gh`, `b_g`**: lineaire lagen en bias voor de gating-functie.
- **`A`**: leerbare parameter die het attractorpunt van de dynamiek bepaalt.
- **`ln_h`**: `LayerNorm` voor stabilisatie van de verborgen toestand.
- **`_last_gate_reg`, `_last_A_reg`**: cache voor de meest recente
  regularisatiewaarden.

### Kernmethoden
- **`forward(x, h, dt)`**: voert één LTC-tijdstap uit met zachte plus/tanh-
  dynamiek en slaat regularisatietermen op.
- **`last_gate_reg` / `last_A_reg`**: eigenschapsmethoden die de laatst
  berekende regularisatietermen teruggeven.

## `PELiNNQEM`
Een liquid neural network-regressor voor quantum error mitigation, opgebouwd uit
één `LTCCell` en een lineaire kop.

### Belangrijkste attributen
- **`cell`**: de onderliggende `LTCCell`.
- **`h0`**: leerbare initiële verborgen toestand.
- **`head`**: lineaire projectie van verborgen toestand naar scalar output.
- **`steps`**: aantal recursieve integratiestappen in de voorwaartse pass.
- **`dt`**: stapgrootte voor de numerieke integratie.
- **`use_tanh_head`**: schakelt optionele `tanh`-activatie op de uitgang in.

### Kernmethoden
- **`forward(x)`**: voert de recurrente dynamiek `steps` keer uit en produceert
  een voorspelling per sample.
- **`compute_loss(pred, target, ...)`**: wikkelt de globale `physics_loss`
  helper zodat invariantie-, Huber- of MSE-verlies eenvoudig toegepast kan
  worden, inclusief toegang tot de laatste regularisatietermen.

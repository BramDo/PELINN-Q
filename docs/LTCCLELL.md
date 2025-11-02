# Understanding `LTCCell`

The `LTCCell` class in `pelinn/model.py` implements a **Liquid Time-Constant (LTC)**
recurrent neural network cell. LTC cells parameterise continuous-time dynamics
and were introduced to model systems with rapidly changing time constants. The
implementation in this repository mirrors the formulation from Hasani et al.
("Liquid Time-constant Networks") and augments it with a couple of hooks for
simple regularisation.

## Weights and learnable parameters

The constructor initialises three groups of learnable components:

1. **Time-constant pathway (`tau`)**
   * `W_tx` – linear map from the input vector \(x_t\).
   * `W_th` – linear map from the previous hidden state \(h_{t-1}\).
   * `b_t` – bias term.
   These tensors produce the pre-activation that is passed through a `softplus`
   to ensure strictly positive time constants.

2. **Gating pathway (`g`)**
   * `W_gx` and `W_gh` – linear maps applied to the input and hidden state.
   * `b_g` – bias term.
   The gate is squashed by `sigmoid`, constraining it to \([0, 1]\).

3. **Attractor vector (`A`)**
   * `A` – learnable anchor point towards which the hidden state is driven.

Finally, a `LayerNorm` module (`ln_h`) stabilises the hidden state after each
Euler step, while the small constant `eps` keeps the time constants numerically
bounded away from zero.

## Forward dynamics

During the forward pass the cell receives the current input `x`, the previous
hidden state `h`, and an integration step `dt` (default `0.25`). The dynamics are
split into three stages:

1. **Time constant:**
   ```python
   tau = F.softplus(self.W_tx(x) + self.W_th(h) + self.b_t) + self.eps
   ```
   The softplus enforces positive time constants and the additional `eps`
   prevents numerical singularities.

2. **Input gate:**
   ```python
   g = torch.sigmoid(self.W_gx(x) + self.W_gh(h) + self.b_g)
   ```
   The gate controls how much the attractor influences the hidden state.

3. **Hidden-state derivative:**
   ```python
   dh = -h / tau + g * (self.A - h)
   ```
   The first term models exponential decay, while the second term pulls the
   state toward `A` proportionally to the gate activation.

The cell integrates these dynamics with an explicit Euler step

```python
h_next = h + dt * dh
return self.ln_h(h_next)
```

giving the next hidden state, normalised for stability.

## Regularisation hooks

For downstream loss functions the cell caches two scalar quantities:

* `last_gate_reg` – mean of \(g (1 - g)\), encouraging non-saturated gates when
  penalised.
* `last_A_reg` – mean squared magnitude of the attractor vector, favouring
  smaller anchors.

These are exposed as read-only properties so that a training loop can include
additional penalties without recomputing the forward pass.

``` mermaid
flowchart TD
        X(["Input features"])
        subgraph LTCCell["LTC Cell"]
            direction TB
            X -->|W_tx, W_th| Tau["Tijdconstante τ"]
            X -->|W_gx, W_gh| Gate["Gate g"]
            Tau --> Euler["Euler update"]
            Gate --> Euler
            A["Attractor A"] --> Euler
            Euler --> LN["LayerNorm"]
        end
        LN --> Repeat{{"Herhaal steps"}}
        Repeat --> Head["Linear head"]
        Head -->|opt. tanh| Output
```


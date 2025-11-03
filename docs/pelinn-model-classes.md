# Classes in `pelinn/model.py`

## `LTCCell`
A `torch.nn.Module` that implements a Liquid Time-Constant (LTC) cell with
regularisation terms for gating activations and the parameter `A`.

### Key attributes
- **`W_tx`, `W_th`, `b_t`**: linear layers and bias for the time-constant
  dynamics.
- **`W_gx`, `W_gh`, `b_g`**: linear layers and bias for the gating function.
- **`A`**: learnable parameter that sets the attractor point of the dynamics.
- **`ln_h`**: `LayerNorm` that stabilises the hidden state.
- **`_last_gate_reg`, `_last_A_reg`**: cache storing the most recent
  regularisation values.

### Core methods
- **`forward(x, h, dt)`**: executes one LTC time step with softplus/tanh
  dynamics and caches regularisation terms.
- **`last_gate_reg` / `last_A_reg`**: property methods returning the last
  computed regularisation values.

## `PELiNNQEM`
A liquid neural network regressor for quantum error mitigation built from a
single `LTCCell` and a linear head.

### Key attributes
- **`cell`**: the underlying `LTCCell`.
- **`h0`**: learnable initial hidden state.
- **`head`**: linear projection from hidden state to scalar output.
- **`steps`**: number of recursive integration steps during the forward pass.
- **`dt`**: step size for numerical integration.
- **`use_tanh_head`**: toggles an optional `tanh` activation on the output.

### Core methods
- **`forward(x)`**: runs the recurrent dynamics `steps` times and returns a
  prediction per sample.
- **`compute_loss(pred, target, ...)`**: wraps the global `physics_loss` helper
  so invariance, Huber, or MSE losses can be applied easily, with access to the
  latest regularisation terms.

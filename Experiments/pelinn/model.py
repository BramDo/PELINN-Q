# 


"""Liquid neural network model used by PE-LiNN QEM experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LTCCell(nn.Module):
    """Liquid Time-Constant (LTC) cell with simple regularisation hooks."""

    def __init__(self, in_dim: int, hid_dim: int, eps: float = 1e-3) -> None:
        super().__init__()
        self.W_tx = nn.Linear(in_dim, hid_dim, bias=False)
        self.W_th = nn.Linear(hid_dim, hid_dim, bias=False)
        self.b_t = nn.Parameter(torch.zeros(hid_dim))
        self.eps = eps

        self.W_gx = nn.Linear(in_dim, hid_dim, bias=False)
        self.W_gh = nn.Linear(hid_dim, hid_dim, bias=False)
        self.b_g = nn.Parameter(torch.zeros(hid_dim))

        self.A = nn.Parameter(torch.zeros(hid_dim))
        self.ln_h = nn.LayerNorm(hid_dim)  # stabilises the hidden state

        self._last_gate_reg: torch.Tensor | None = None
        self._last_A_reg: torch.Tensor | None = None

    @property
    def last_gate_reg(self) -> torch.Tensor | None:
        return self._last_gate_reg

    @property
    def last_A_reg(self) -> torch.Tensor | None:
        return self._last_A_reg

    def forward(self, x: torch.Tensor, h: torch.Tensor, dt: float = 0.25) -> torch.Tensor:
        # Hasani-style dynamics
        tau = F.softplus(self.W_tx(x) + self.W_th(h) + self.b_t) + self.eps
        g = torch.sigmoid(self.W_gx(x) + self.W_gh(h) + self.b_g)
        dh = -h / tau + g * (self.A - h)

        # Cache regularisation terms for later use
        self._last_gate_reg = (g * (1.0 - g)).mean()
        self._last_A_reg = (self.A ** 2).mean()

        h_next = h + dt * dh
        return self.ln_h(h_next)  # mild normalisation helps optimisation


class PELiNNQEM(nn.Module):
    """Liquid neural network regressor for quantum error mitigation."""

    def __init__(
        self,
        in_dim: int,
        hid_dim: int = 96,
        steps: int = 6,
        dt: float = 0.25,
        use_tanh_head: bool = False,
    ) -> None:
        super().__init__()
        self.cell = LTCCell(in_dim, hid_dim)
        self.h0 = nn.Parameter(torch.zeros(hid_dim))
        self.head = nn.Linear(hid_dim, 1)
        self.steps = steps
        self.dt = dt
        self.use_tanh_head = use_tanh_head

    @property
    def last_gate_reg(self) -> torch.Tensor | None:
        return self.cell.last_gate_reg

    @property
    def last_A_reg(self) -> torch.Tensor | None:
        return self.cell.last_A_reg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        h = self.h0.unsqueeze(0).expand(batch, -1)
        for _ in range(self.steps):
            h = self.cell(x, h, dt=self.dt)
        y = self.head(h).squeeze(-1)
        return torch.tanh(y) if self.use_tanh_head else y

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        groups: list[list[int]] | None = None,
        alpha_inv: float = 0.1,
        loss_type: str = "mse",
        huber_beta: float = 0.1,
    ) -> torch.Tensor:
        return physics_loss(
            pred,
            target,
            groups=groups,
            alpha_inv=alpha_inv,
            reg_gate=self.last_gate_reg,
            reg_A=self.last_A_reg,
            loss_type=loss_type,
            huber_beta=huber_beta,
        )


def physics_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    groups: list[list[int]] | None = None,
    alpha_inv: float = 0.1,
    reg_gate: torch.Tensor | None = None,
    reg_A: torch.Tensor | None = None,
    loss_type: str = "mse",
    huber_beta: float = 0.1,
) -> torch.Tensor:
    """Combine data MSE with invariance & optional regularisers."""

    if loss_type == "huber":
        loss = F.smooth_l1_loss(pred, target, beta=huber_beta)
    else:
        loss = F.mse_loss(pred, target)

    if groups:
        inv = 0.0
        cnt = 0
        for idxs in groups:
            if len(idxs) < 2:
                continue
            p = pred[idxs]
            inv += (p.unsqueeze(0) - p.unsqueeze(1)).abs().mean()
            cnt += 1
        inv = inv / max(cnt, 1)
        loss = loss + alpha_inv * inv

    if reg_gate is not None:
        loss = loss + 1e-3 * reg_gate
    if reg_A is not None:
        loss = loss + 1e-3 * reg_A

    return loss
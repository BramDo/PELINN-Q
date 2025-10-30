# scripts/train_pelinn.py
import torch, numpy as np, random
from torch.utils.data import Dataset, DataLoader
from pelinn.model import PELiNNQEM, physics_loss

class QemDataset(Dataset):
    def __init__(self, samples):
        self.X = np.stack([s.x for s in samples]).astype(np.float32)
        self.y = np.array([s.y_ideal for s in samples], dtype=np.float32)
        self.cid = np.array(
            [
                int(s.meta.get("circuit_index", id(s.meta["qc"])))
                for s in samples
            ],
            dtype=np.int64,
        )  # circuit id for grouping
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i], self.cid[i]

def make_groups(cids):
    # group indices sharing the same circuit id
    groups = {}
    for i, c in enumerate(cids): groups.setdefault(int(c), []).append(i)
    return list(groups.values())

def train(model, loader, opt, device="cpu"):
    model.train()
    for X, y, cid in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        groups = make_groups(cid.tolist())
        loss = physics_loss(
            pred,
            y,
            groups,
            alpha_inv=0.1,
            reg_gate=model.last_gate_reg,
            reg_A=model.last_A_reg,
        )
        opt.zero_grad(); loss.backward(); opt.step()

# usage sketch
samples = synthesize_samples(circuits, observables, noise_grid)
# ds = QemDataset(samples); dl = DataLoader(ds, batch_size=128, shuffle=True)
# model = PELiNNQEM(in_dim=ds.X.shape[1]).to("cuda" if torch.cuda.is_available() else "cpu")
# opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
# for epoch in range(100): train(model, dl, opt)

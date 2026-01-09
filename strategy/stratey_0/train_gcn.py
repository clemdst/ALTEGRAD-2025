import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_add_pool

from utils.data_utils import (
    load_id2emb,
    PreprocessedGraphDataset, collate_fn
)


# =========================================================
# CONFIG
# =========================================================
# Data paths
TRAIN_GRAPHS = "data/train_graphs.pkl"
VAL_GRAPHS   = "data/validation_graphs.pkl"
TEST_GRAPHS  = "data/test_graphs.pkl"

TRAIN_EMB_CSV = "data/train_embeddings.csv"
VAL_EMB_CSV   = "data/validation_embeddings.csv"

# Training parameters
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_MODE = True
N_SAMPLES = 10


# =========================================================
# MODEL: GNN to encode graphs (simple GCN, no edge features)
# =========================================================
class MolGNN(nn.Module):
    def __init__(self, hidden=128, out_dim=256, layers=3):
        super().__init__()

        # Use a single learnable embedding for all nodes (no node features)
        self.node_init = nn.Parameter(torch.randn(hidden))

        self.convs = nn.ModuleList()
        for _ in range(layers):
            self.convs.append(GCNConv(hidden, hidden))

        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, batch: Batch):
        # Initialize all nodes with the same learnable embedding
        num_nodes = batch.x.size(0)
        h = self.node_init.unsqueeze(0).expand(num_nodes, -1)
        
        for conv in self.convs:
            h = conv(h, batch.edge_index)
            h = F.relu(h)
        g = global_add_pool(h, batch.batch)
        g = self.proj(g)
        g = F.normalize(g, dim=-1)
        return g
    

# =========================================================
# Limit training size
# =========================================================
    
class LimitedGraphDataset:
    """Wrapper to limit the number of samples from a dataset"""
    def __init__(self, dataset, n_samples=None):
        self.dataset = dataset
        self.n_samples = n_samples if n_samples is not None else len(dataset)
        self.n_samples = min(self.n_samples, len(dataset))
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        if idx >= self.n_samples:
            raise IndexError("Index out of range")
        return self.dataset[idx]


# =========================================================
# Training and Evaluation
# =========================================================
def train_epoch(mol_enc, loader, optimizer, device):
    mol_enc.train()

    total_loss, total = 0.0, 0
    for graphs, text_emb in loader:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)

        mol_vec = mol_enc(graphs)
        txt_vec = F.normalize(text_emb, dim=-1)

        loss = F.mse_loss(mol_vec, txt_vec)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = graphs.num_graphs
        total_loss += loss.item() * bs
        total += bs

    return total_loss / total


@torch.no_grad()
def eval_retrieval(data_path, emb_dict, mol_enc, device):
    ds = PreprocessedGraphDataset(data_path, emb_dict)
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    all_mol, all_txt = [], []
    for graphs, text_emb in dl:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)
        all_mol.append(mol_enc(graphs))
        all_txt.append(F.normalize(text_emb, dim=-1))
    all_mol = torch.cat(all_mol, dim=0)
    all_txt = torch.cat(all_txt, dim=0)

    sims = all_txt @ all_mol.t()
    ranks = sims.argsort(dim=-1, descending=True)

    N = all_txt.size(0)
    device = sims.device
    correct = torch.arange(N, device=device)

    pos = (ranks == correct.unsqueeze(1)).nonzero()[:, 1] + 1

    mrr = (1.0 / pos.float()).mean().item()

    results = {"MRR": mrr}

    for k in (1, 5, 10):
        hitk = (pos <= k).float().mean().item()
        results[f"R@{k}"] = hitk
        results[f"Hit@{k}"] = hitk

    return results

@torch.no_grad()
def eval_retrieval_test(loader, mol_enc, device):
    mol_enc.eval()
    
    all_mol, all_txt = [], []
    for graphs, text_emb in loader:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)
        all_mol.append(mol_enc(graphs))
        all_txt.append(F.normalize(text_emb, dim=-1))
    
    if not all_mol:  # Empty loader
        return {}
        
    all_mol = torch.cat(all_mol, dim=0)
    all_txt = torch.cat(all_txt, dim=0)

    sims = all_txt @ all_mol.t()
    ranks = sims.argsort(dim=-1, descending=True)

    N = all_txt.size(0)
    device = sims.device
    correct = torch.arange(N, device=device)

    pos = (ranks == correct.unsqueeze(1)).nonzero()[:, 1] + 1

    mrr = (1.0 / pos.float()).mean().item()

    results = {"MRR": mrr}

    for k in (1, 5, 10):
        hitk = (pos <= k).float().mean().item()
        results[f"R@{k}"] = hitk
        results[f"Hit@{k}"] = hitk

    return results

# =========================================================
# Main Training Loop
# =========================================================
def main():
    print(f"Device: {DEVICE}")

    if TEST_MODE:
        print(f"Running in TEST MODE with {N_SAMPLES} samples")


    train_emb = load_id2emb(TRAIN_EMB_CSV)
    val_emb = load_id2emb(VAL_EMB_CSV) if os.path.exists(VAL_EMB_CSV) else None

    emb_dim = len(next(iter(train_emb.values())))

    if not os.path.exists(TRAIN_GRAPHS):
        print(f"Error: Preprocessed graphs not found at {TRAIN_GRAPHS}")
        print("Please run: python prepare_graph_data.py")
        return
    
    full_train_ds = PreprocessedGraphDataset(TRAIN_GRAPHS, train_emb)

    if TEST_MODE:
        train_ds = LimitedGraphDataset(full_train_ds, N_SAMPLES)
        print(f"Limited training dataset to {len(train_ds)} samples")
    else:
        train_ds = full_train_ds
        print(f"Using full training dataset with {len(train_ds)} samples")
    


    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    mol_enc = MolGNN(out_dim=emb_dim).to(DEVICE)

    optimizer = torch.optim.Adam(mol_enc.parameters(), lr=LR)

    for ep in range(EPOCHS):
        train_loss = train_epoch(mol_enc, train_dl, optimizer, DEVICE)
        if val_emb is not None and os.path.exists(VAL_GRAPHS):
            if TEST_MODE:
                # Also limit validation for consistency
                full_val_ds = PreprocessedGraphDataset(VAL_GRAPHS, val_emb)
                limited_val_ds = LimitedGraphDataset(full_val_ds, N_SAMPLES//2)  # Use fewer validation samples
                val_dl = DataLoader(limited_val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
                val_scores = eval_retrieval_test(val_dl, mol_enc, DEVICE)
            else:
                val_scores = eval_retrieval(VAL_GRAPHS, val_emb, mol_enc, DEVICE)
        else:
            val_scores = {}
        print(f"Epoch {ep+1}/{EPOCHS} - loss={train_loss:.4f} - val={val_scores}")
    
    model_path = "model_checkpoint.pt"
    torch.save(mol_enc.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()

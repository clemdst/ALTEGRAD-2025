import os
import copy
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.nn import GINEConv, global_add_pool

from utils.data_utils import (
    load_id2emb,
    PreprocessedGraphDataset, collate_fn,
    x_map, 
    e_map
)

# =========================================================
# CONFIG
# =========================================================
# Data paths    
TRAIN_GRAPHS = "./data/train_graphs.pkl"
VAL_GRAPHS   = "./data/validation_graphs.pkl"
TEST_GRAPHS  = "./data/test_graphs.pkl"

TRAIN_EMB_CSV = "embeddings/train_scibert_embeddings.csv"
VAL_EMB_CSV   = "embeddings/validation_scibert_embeddings.csv"

# Training parameters
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_MODE = False
N_SAMPLES = 10

# --- NEW: Matryoshka Dimensions ---
# These are the "slices" we want to optimize for. 
# The model will learn to be accurate at 64 dims, 128 dims, etc.
MATRYOSHKA_DIMS = [64, 128, 256, 512, 768] 


# =========================================================
# MODEL: GNN to encode graphs (simple GCN, no edge features)
# =========================================================
class MolGINE(nn.Module):
    def __init__(self, hidden=128, out_dim=256, layers=3):
        super().__init__()
        
        #Embed node features
        self.node_emb=nn.ModuleList()
        for key in x_map.keys():
            self.node_emb.append(nn.Embedding(len(x_map[key]), hidden))
        
        self.node_proj=nn.Linear(hidden*len(x_map),hidden)    
            
        #Embed edge features
        self.edge_emb=nn.ModuleList()
        for key in e_map.keys():
            self.edge_emb.append(nn.Embedding(len(e_map[key]), hidden))
        
        self.edge_proj=nn.Linear(hidden*len(e_map),hidden)
        
        #GINE Layers
        self.convs = nn.ModuleList()
        for _ in range(layers):
            mlp=nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden)
            )
            self.convs.append(GINEConv(mlp))
        
        self.proj = nn.Linear(hidden, out_dim)


    def forward(self, batch: Batch):
        #Embed node features
        node_feats=[]
        for i,emb in enumerate(self.node_emb):
            node_feats.append(emb(batch.x[:,i]))
        
     
        x=torch.cat(node_feats,dim=-1)
        x=self.node_proj(x)

        #Embed edge features
        edge_feats=[]
        for i,emb in enumerate(self.edge_emb):
            edge_feats.append(emb(batch.edge_attr[:,i]))
        
        edge_feats=torch.cat(edge_feats,dim=-1)
        edge_feats=self.edge_proj(edge_feats)
        
        #message passing
        for conv in self.convs:
            x=conv(x,batch.edge_index,edge_feats)
            x=F.relu(x)
            
        #output projection
        g = global_add_pool(x, batch.batch)
        g = self.proj(g)
        
        # --- NOTE: Removed F.normalize here ---
        # In Matryoshka training, we must normalize AFTER slicing, 
        # not before. We return the raw projected vector.
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
    
    # Weights for each Matryoshka dimension (optional, usually all 1.0)
    # You can set the full dim to have higher weight if that's the priority.
    loss_weights = [1.0] * len(MATRYOSHKA_DIMS)

    for graphs, text_emb in loader:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)

        # 1. Get RAW (un-normalized) vectors
        mol_raw = mol_enc(graphs)
        txt_raw = text_emb # SciBERT is usually un-normalized raw output

        batch_loss = 0.0
        
        # --- NEW: Matryoshka Loop ---
        # We calculate the loss at every "slice" defined in MATRYOSHKA_DIMS
        for i, dim in enumerate(MATRYOSHKA_DIMS):
            # Ensure we don't slice more than available (safety check)
            if dim > mol_raw.size(1): break
            
            # A. Slice to the current dimension
            mol_slice = mol_raw[:, :dim]
            txt_slice = txt_raw[:, :dim]
            
            # B. Normalize the SLICE (Crucial!)
            mol_norm = F.normalize(mol_slice, p=2, dim=-1)
            txt_norm = F.normalize(txt_slice, p=2, dim=-1)
            
            # C. Calculate Similarity (InfoNCE)
            logits = torch.matmul(mol_norm, txt_norm.T) / 0.07
            labels = torch.arange(logits.size(0)).to(device)
            
            loss_i = F.cross_entropy(logits, labels)
            loss_t = F.cross_entropy(logits.T, labels)
            current_loss = (loss_i + loss_t) / 2
            
            # D. Accumulate
            batch_loss += current_loss * loss_weights[i]

        # Average the loss across dimensions so magnitudes don't explode
        batch_loss = batch_loss / len(MATRYOSHKA_DIMS)
        
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        bs = graphs.num_graphs
        total_loss += batch_loss.item() * bs
        total += bs

    return total_loss / total


@torch.no_grad()
def eval_retrieval(data_path, emb_dict, mol_enc, device):
    """
    Evaluates MRR at ALL Matryoshka dimensions to verify efficient retrieval.
    """
    ds = PreprocessedGraphDataset(data_path, emb_dict)
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    all_mol_raw, all_txt_raw = [], []
    for graphs, text_emb in dl:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)
        all_mol_raw.append(mol_enc(graphs))
        all_txt_raw.append(text_emb)
        
    all_mol_raw = torch.cat(all_mol_raw, dim=0)
    all_txt_raw = torch.cat(all_txt_raw, dim=0)

    results = {}
    
    # --- NEW: Evaluate at every dimension ---
    print("\n--- Matryoshka Evaluation ---")
    for dim in MATRYOSHKA_DIMS:
        if dim > all_mol_raw.size(1): continue
        
        # Slice and Normalize
        mol_slice = F.normalize(all_mol_raw[:, :dim], p=2, dim=-1)
        txt_slice = F.normalize(all_txt_raw[:, :dim], p=2, dim=-1)
        
        # Standard Retrieval Metric Calculation
        sims = txt_slice @ mol_slice.t()
        ranks = sims.argsort(dim=-1, descending=True)
        N = txt_slice.size(0)
        correct = torch.arange(N, device=device)
        pos = (ranks == correct.unsqueeze(1)).nonzero()[:, 1] + 1
        mrr = (1.0 / pos.float()).mean().item()
        
        results[f"MRR@{dim}"] = mrr
        print(f"Dim {dim}: MRR = {mrr:.4f}")

    # Return the MRR of the largest dimension as the primary metric for early stopping
    max_dim = MATRYOSHKA_DIMS[-1]
    results['MRR'] = results[f"MRR@{max_dim}"]
    
    return results

@torch.no_grad()
def eval_retrieval_test(loader, mol_enc, device):
    """
    Same as above but accepts a loader directly (for validation during training).
    """
    mol_enc.eval()
    
    all_mol_raw, all_txt_raw = [], []
    for graphs, text_emb in loader:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)
        all_mol_raw.append(mol_enc(graphs))
        all_txt_raw.append(text_emb)
    
    if not all_mol_raw: return {}
        
    all_mol_raw = torch.cat(all_mol_raw, dim=0)
    all_txt_raw = torch.cat(all_txt_raw, dim=0)
    
    results = {}
    
    # Just calculate max dim MRR to keep log output clean during training loop,
    # or calculate all if you want verbose logs. Here we do Max Dim only.
    dim = MATRYOSHKA_DIMS[-1]
    
    mol_slice = F.normalize(all_mol_raw[:, :dim], p=2, dim=-1)
    txt_slice = F.normalize(all_txt_raw[:, :dim], p=2, dim=-1)

    sims = txt_slice @ mol_slice.t()
    ranks = sims.argsort(dim=-1, descending=True)

    N = txt_slice.size(0)
    correct = torch.arange(N, device=sims.device)

    pos = (ranks == correct.unsqueeze(1)).nonzero()[:, 1] + 1
    mrr = (1.0 / pos.float()).mean().item()

    results["MRR"] = mrr
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

    # Ensure emb_dim covers our largest Matryoshka dimension
    emb_dim = len(next(iter(train_emb.values())))
    assert emb_dim >= max(MATRYOSHKA_DIMS), f"Embedding size {emb_dim} is smaller than max Matryoshka dim {max(MATRYOSHKA_DIMS)}"

    if not os.path.exists(TRAIN_GRAPHS):
        print(f"Error: Preprocessed graphs not found at {TRAIN_GRAPHS}")
        return
    
    full_train_ds = PreprocessedGraphDataset(TRAIN_GRAPHS, train_emb)

    if TEST_MODE:
        train_ds = LimitedGraphDataset(full_train_ds, N_SAMPLES)
        print(f"Limited training dataset to {len(train_ds)} samples")
    else:
        train_ds = full_train_ds
        print(f"Using full training dataset with {len(train_ds)} samples")
    

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    mol_enc = MolGINE(out_dim=emb_dim).to(DEVICE)

    optimizer = torch.optim.Adam(mol_enc.parameters(), lr=LR)

    best_mrr = 0.0
    best_model_weights = None
    best_model_val_score=None

    for ep in range(EPOCHS):
        train_loss = train_epoch(mol_enc, train_dl, optimizer, DEVICE)
        
        # Validation Logic
        if val_emb is not None and os.path.exists(VAL_GRAPHS):
            # For validation logs, we usually just want to see if the full model is improving
            if TEST_MODE:
                full_val_ds = PreprocessedGraphDataset(VAL_GRAPHS, val_emb)
                limited_val_ds = LimitedGraphDataset(full_val_ds, N_SAMPLES//2) 
                val_dl = DataLoader(limited_val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
                val_scores = eval_retrieval_test(val_dl, mol_enc, DEVICE)
            else:
                # We use the detailed eval_retrieval here to see all dimensions!
                val_scores = eval_retrieval(VAL_GRAPHS, val_emb, mol_enc, DEVICE)
            
            current_mrr = val_scores['MRR']
        else:
            val_scores = {}
            current_mrr = 0

        print(f"Epoch {ep+1}/{EPOCHS} - loss={train_loss:.4f} - Full MRR={current_mrr:.4f}")
        
        if current_mrr > best_mrr:
            best_mrr = current_mrr
            best_model_val_score=val_scores
            if best_model_weights is not None:
                del best_model_weights
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            best_model_weights = copy.deepcopy(mol_enc.state_dict())
            print(f"New best found!")

    model_path = "models/model_strategy_3.pt"
    if best_model_weights:
        torch.save(best_model_weights, model_path)
    else:
        torch.save(mol_enc.state_dict(), model_path)
        
    print(f"\nModel saved to {model_path}")
    print(f"Final Best Scores: ", best_model_val_score)


if __name__ == "__main__":
    main()

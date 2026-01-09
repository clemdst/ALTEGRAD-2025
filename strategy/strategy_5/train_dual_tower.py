import os
import copy
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from torch_geometric.data import Batch
from torch_geometric.nn import TransformerConv, global_add_pool, global_mean_pool

from utils.data_utils import (
    load_id2emb, load_descriptions_from_graphs,
    PreprocessedGraphDataset, collate_fn,
    x_map, e_map
)

# =========================================================
# CONFIGURATION
# =========================================================
TRAIN_GRAPHS = "./data/train_graphs.pkl"
VAL_GRAPHS   = "./data/validation_graphs.pkl"
TEST_GRAPHS  = "./data/test_graphs.pkl"

# Using your specific SciBERT embeddings
TRAIN_EMB_CSV = "embeddings/train_scibert_embeddings.csv"
VAL_EMB_CSV   = "embeddings/validation_scibert_embeddings.csv"

# Output Paths
MODEL_PATH = "models/model_strategy_5.pt"
# Training Settings
# Set TRAIN_FULL_DATA = True for your FINAL run (uses Train + Val)
# Set TRAIN_FULL_DATA = False to monitor validation score first
TRAIN_FULL_DATA = False 

BATCH_SIZE = 24       
EPOCHS = 25           
LR = 2e-4             
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# THE DUAL TOWER MODEL
# =========================================================
class MolTransformerDual(nn.Module):
    def __init__(self, hidden=128, text_dim=768, out_dim=768, layers=3, heads=4):
        super().__init__()
        
        # --- TOWER A: GRAPH ENCODER (Transformer) ---
        self.node_emb = nn.ModuleList([nn.Embedding(len(x_map[key]), hidden) for key in x_map])
        self.node_proj = nn.Linear(hidden * len(x_map), hidden)
        
        self.edge_emb = nn.ModuleList([nn.Embedding(len(e_map[key]), hidden) for key in e_map])
        self.edge_proj = nn.Linear(hidden * len(e_map), hidden)
        
        self.convs = nn.ModuleList()
        for _ in range(layers):
            self.convs.append(TransformerConv(
                in_channels=hidden, out_channels=hidden//heads, heads=heads,
                dropout=0.1, edge_dim=hidden, beta=True
            ))
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        
        # Graph Projection Head
        self.graph_proj = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden * 2, out_dim)
        )

        # --- TOWER B: TEXT ENCODER (The "Adapter") ---
        # Takes fixed SciBERT and learns to map it to Chemistry Space
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.BatchNorm1d(text_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(text_dim, out_dim)
        )
        
        # Init text projection close to identity to start stable
        nn.init.eye_(self.text_proj[0].weight)
        nn.init.eye_(self.text_proj[4].weight)

    def forward_graph(self, batch):
        """Pass Graph through Tower A"""
        node_feats = [emb(batch.x[:, i]) for i, emb in enumerate(self.node_emb)]
        x = self.node_proj(torch.cat(node_feats, dim=-1))
        
        edge_feats = [emb(batch.edge_attr[:, i]) for i, emb in enumerate(self.edge_emb)]
        edge_attr = self.edge_proj(torch.cat(edge_feats, dim=-1))
        
        for conv, ln in zip(self.convs, self.layer_norms):
            x = F.relu(ln(x + conv(x, batch.edge_index, edge_attr)))
            
        g = global_add_pool(x, batch.batch) + global_mean_pool(x, batch.batch)
        return self.graph_proj(g)

    def forward_text(self, text_emb):
        """Pass Text through Tower B"""
        return self.text_proj(text_emb)

    def forward(self, batch, text_emb):
        """Training Step: Return both vectors"""
        g_vec = F.normalize(self.forward_graph(batch), p=2, dim=-1)
        t_vec = F.normalize(self.forward_text(text_emb), p=2, dim=-1)
        return g_vec, t_vec

# =========================================================
# TRAINING UTILS
# =========================================================
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total = 0.0, 0
    
    for graphs, text_emb in loader:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)
        
        # 1. Forward both towers
        g_vec, t_vec = model(graphs, text_emb)
        
        # 2. Symmetric Contrastive Loss
        logits = (g_vec @ t_vec.T) / 0.07
        labels = torch.arange(logits.size(0)).to(device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * graphs.num_graphs
        total += graphs.num_graphs
    return total_loss / total

@torch.no_grad()
def eval_retrieval(loader, model, device):
    """Validation Helper"""
    model.eval()
    all_g, all_t = [], []
    for graphs, text_emb in loader:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)
        all_g.append(F.normalize(model.forward_graph(graphs), dim=-1))
        all_t.append(F.normalize(model.forward_text(text_emb), dim=-1))
        
    if not all_g: return {}
    all_g = torch.cat(all_g, 0)
    all_t = torch.cat(all_t, 0)
    
    sims = all_t @ all_g.t()
    ranks = sims.argsort(dim=-1, descending=True)
    correct = torch.arange(all_t.size(0), device=device)
    pos = (ranks == correct.unsqueeze(1)).nonzero()[:, 1] + 1
    return {"MRR": (1.0/pos.float()).mean().item()}

# =========================================================
# MAIN EXECUTION
# =========================================================
def main():
    print(f"Device: {DEVICE}")
    print(f"Training Mode: {'FULL DATA (Train+Val)' if TRAIN_FULL_DATA else 'VALIDATION SPLIT'}")

    # 1. Load Embeddings
    if not os.path.exists(TRAIN_EMB_CSV):
        print(f"Error: Embedding file not found at {TRAIN_EMB_CSV}")
        return

    print("Loading Embeddings...")
    train_emb = load_id2emb(TRAIN_EMB_CSV)
    val_emb = load_id2emb(VAL_EMB_CSV) if os.path.exists(VAL_EMB_CSV) else {}
    
    # 2. Prepare Datasets
    print("Loading Graph Datasets...")
    train_ds_raw = PreprocessedGraphDataset(TRAIN_GRAPHS, train_emb)
    val_ds_raw = PreprocessedGraphDataset(VAL_GRAPHS, val_emb) if val_emb else None

    # 3. Setup DataLoaders
    if TRAIN_FULL_DATA and val_ds_raw:
        # MERGE DATASETS (Train + Val) for final model
        full_dataset = ConcatDataset([train_ds_raw, val_ds_raw])
        train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = None
        print(f"Dataset Merged: {len(full_dataset)} total samples.")
    else:
        # Standard Split
        train_loader = DataLoader(train_ds_raw, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds_raw, batch_size=32, shuffle=False, collate_fn=collate_fn) if val_ds_raw else None
        print(f"Train samples: {len(train_ds_raw)}")
        if val_ds_raw: print(f"Val samples:   {len(val_ds_raw)}")

    # 4. Model Setup
    # Assumes embeddings are 768 dim (SciBERT)
    model = MolTransformerDual(hidden=128, text_dim=768, out_dim=768).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 5. Training Loop
    best_mrr = 0.0
    
    print(f"\n--- Starting Dual Tower Training ({EPOCHS} Epochs) ---")
    for ep in range(EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, DEVICE)
        
        # Validation Logic
        if val_loader:
            val_scores = eval_retrieval(val_loader, model, DEVICE)
            print(f"Epoch {ep+1}/{EPOCHS} | Loss: {loss:.4f} | MRR: {val_scores.get('MRR', 0):.4f}")
            
            # Save Best Model
            if val_scores.get('MRR', 0) > best_mrr:
                best_mrr = val_scores['MRR']
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"  >>> New Best Model Saved (MRR: {best_mrr:.4f})")
        else:
            # Blind Training (Full Data) - Just save the latest
            print(f"Epoch {ep+1}/{EPOCHS} | Loss: {loss:.4f}")
            torch.save(model.state_dict(), MODEL_PATH)
            
        scheduler.step()

    print(f"\nTraining Complete. Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()

# %load /kaggle/working/advanced_project_clone-/model/train_dual_tower_reranker.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import random
import os

# Reuse your existing data utils
from utils.data_utils import (
    PreprocessedGraphDataset, collate_fn, load_id2emb
)
# Reuse the model class from Phase 6 (ensure train.py is in the same folder)
from strategy.strategy_5.train_dual_tower import MolTransformerDual 

# =========================================================
# CONFIGURATION
# =========================================================
# Input Data Paths
TRAIN_GRAPHS = "./data/train_graphs.pkl"
VAL_GRAPHS   = "./data/validation_graphs.pkl"
TEST_GRAPHS  = "./data/test_graphs.pkl"

# Using your specific SciBERT embeddings
TRAIN_EMB_CSV = "embeddings/train_chembed_embeddings.csv"
VAL_EMB_CSV   = "embeddings/validation_chembed_embeddings.csv"

# Model Paths
MODEL_PATH = "models/model_strategy_6.pt"         # The frozen backbone (Input)
RERANKER_OUTPUT_PATH = "models/model_strategy_7.pt" # The new reranker (Output)
# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-3
HIDDEN_DIM = 128 # Must match the backbone hidden dim
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# 1. THE RERANKER MODEL (Binary Classifier)
# =========================================================
class RerankNet(nn.Module):
    def __init__(self, input_dim=768*2): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Output: Logit
        )
        
    def forward(self, g_emb, t_emb):
        # Concatenate: [Batch, 768] + [Batch, 768] -> [Batch, 1536]
        combined = torch.cat([g_emb, t_emb], dim=1)
        return self.net(combined)

# =========================================================
# 2. DATASET FOR RERANKING
# =========================================================
class RerankDataset(Dataset):
    def __init__(self, g_vecs, t_vecs):
        self.g_vecs = g_vecs
        self.t_vecs = t_vecs
        self.num_samples = len(g_vecs)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Positive Pair (Ground Truth)
        g_emb = self.g_vecs[idx]
        t_pos = self.t_vecs[idx]
        
        # 2. Negative Pair (Random Hard Negative)
        # We pick a random text that IS NOT the ground truth
        neg_idx = random.randint(0, self.num_samples - 1)
        while neg_idx == idx:
            neg_idx = random.randint(0, self.num_samples - 1)
        t_neg = self.t_vecs[neg_idx]
        
        return g_emb, t_pos, t_neg

# =========================================================
# 3. HELPER FUNCTIONS
# =========================================================
def precompute_embeddings(backbone, loader, device, desc="Pre-computing"):
    """Pass data through frozen Phase 6 model once to speed up training."""
    backbone.eval()
    print(f"{desc}...")
    
    g_list, t_list = [], []
    
    with torch.no_grad():
        for graphs, text_emb in tqdm(loader):
            graphs = graphs.to(device)
            text_emb = text_emb.to(device)
            
            # Get normalized vectors from the backbone
            g = F.normalize(backbone.forward_graph(graphs), dim=-1)
            t = F.normalize(backbone.forward_text(text_emb), dim=-1)
            
            g_list.append(g.cpu())
            t_list.append(t.cpu())
            
    return torch.cat(g_list, 0), torch.cat(t_list, 0)

def evaluate_reranker(model, loader, device):
    """
    Checks Pairwise Accuracy: 
    Does the Reranker assign a higher score to the True Pair than the Negative Pair?
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for g_emb, t_pos, t_neg in loader:
            g_emb = g_emb.to(device)
            t_pos = t_pos.to(device)
            t_neg = t_neg.to(device)
            
            # Get scores (logits)
            score_pos = model(g_emb, t_pos)
            score_neg = model(g_emb, t_neg)
            
            # Accuracy: Is Positive Score > Negative Score?
            wins = (score_pos > score_neg).float()
            correct += wins.sum().item()
            total += g_emb.size(0)
            
    return correct / total

# =========================================================
# 4. MAIN LOOP
# =========================================================
def main():
    print(f"--- Training Phase 7 Reranker ---")
    print(f"Device: {DEVICE}")
    
    # ---------------- Setup Backbone ----------------
    print(f"Loading Backbone from {MODEL_PATH}...")
    backbone = MolTransformerDual(hidden=HIDDEN_DIM).to(DEVICE)
    
    # Load weights safely (CPU/GPU)
    map_loc = "cpu" if DEVICE == "cpu" else None
    backbone.load_state_dict(torch.load(MODEL_PATH, map_location=map_loc))
    backbone.eval() # Freeze the backbone!
    
    # ---------------- Prepare Data ----------------
    # 1. Load Input Embeddings
    train_embs_raw = load_id2emb(TRAIN_EMB_CSV)
    val_embs_raw   = load_id2emb(VAL_EMB_CSV)
    
    # 2. Load Graph Datasets
    print("Loading Graph Data...")
    ds_train_raw = PreprocessedGraphDataset(TRAIN_GRAPHS, train_embs_raw)
    ds_val_raw   = PreprocessedGraphDataset(VAL_GRAPHS, val_embs_raw)
    
    loader_train_raw = DataLoader(ds_train_raw, batch_size=128, shuffle=False, collate_fn=collate_fn)
    loader_val_raw   = DataLoader(ds_val_raw, batch_size=128, shuffle=False, collate_fn=collate_fn)
    
    # 3. Precompute Vectors (Optimization)
    train_g, train_t = precompute_embeddings(backbone, loader_train_raw, DEVICE, "Processing Train")
    val_g, val_t     = precompute_embeddings(backbone, loader_val_raw, DEVICE, "Processing Val")
    
    # 4. Create Reranking Datasets
    train_ds = RerankDataset(train_g, train_t)
    val_ds   = RerankDataset(val_g, val_t)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # ---------------- Setup Training ----------------
    model = RerankNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    
    best_acc = 0.0
    
    # ---------------- Training Loop ----------------
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for g_emb, t_pos, t_neg in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            g_emb, t_pos, t_neg = g_emb.to(DEVICE), t_pos.to(DEVICE), t_neg.to(DEVICE)
            
            # Positive pass (Target 1)
            pred_pos = model(g_emb, t_pos)
            loss_pos = criterion(pred_pos, torch.ones_like(pred_pos))
            
            # Negative pass (Target 0)
            pred_neg = model(g_emb, t_neg)
            loss_neg = criterion(pred_neg, torch.zeros_like(pred_neg))
            
            loss = loss_pos + loss_neg
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # Validate
        val_acc = evaluate_reranker(model, val_loader, DEVICE)
        
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val Pairwise Acc: {val_acc:.4f}")
        
        # Save Best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), RERANKER_OUTPUT_PATH)
            print(f"  >>> New Best Reranker Saved to {RERANKER_OUTPUT_PATH} (Acc: {best_acc:.4f})")
            
    print(f"Done. Best Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()

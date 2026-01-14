import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm
from strategy.strategy_6.train_dual_tower import MolTransformerDual

from utils.data_utils import (
    load_id2emb, PreprocessedGraphDataset, collate_fn, x_map, e_map
)

# =========================================================
# 1. HARD NEGATIVE SAMPLER
# =========================================================
class HardNegativeSampler(Sampler):
    def __init__(self, data_source, batch_size, hard_neg_path):
        self.n_samples = len(data_source)
        self.batch_size = batch_size
        
        # Load the pre-computed matrix
        self.hard_neg_indices = np.load(hard_neg_path)
        
        # Determine the valid range to avoid IndexError [cite: 31, 32]
        # We only consider indices that exist in both the dataset and the matrix
        self.max_valid_idx = self.hard_neg_indices.shape[0]
        self.available_indices = np.arange(self.max_valid_idx)

    def __iter__(self):
        # Only shuffle indices that we actually have hard negatives for
        indices = torch.randperm(self.max_valid_idx).tolist()
        
        for i in range(0, len(indices), self.batch_size):
            batch = []
            num_anchors = self.batch_size // 2
            anchors = indices[i : i + num_anchors]
            
            for idx in anchors:
                batch.append(idx)
                # Ensure the sampled hard_idx is also within the valid dataset range
                hard_idx = int(np.random.choice(self.hard_neg_indices[idx]))
                if hard_idx < self.n_samples:
                    batch.append(hard_idx)
                else:
                    # Fallback to a random valid index if the hard_idx is out of bounds
                    batch.append(np.random.randint(0, self.max_valid_idx))
            
            yield batch[:self.batch_size]

    def __len__(self):
        # Adjusted length based on valid samples
        return self.max_valid_idx // self.batch_size
# =========================================================
# 2. VALIDATION ENGINE (MRR)
# =========================================================
@torch.no_grad()
def evaluate_retrieval(model, loader, device):
    """
    Evaluates the model on validation graphs by calculating 
    Mean Reciprocal Rank (MRR) for the retrieval task[cite: 80].
    """
    model.eval()
    all_g, all_t = [], []
    for graphs, text_emb in loader:
        graphs, text_emb = graphs.to(device), text_emb.to(device)
        # Normalize vectors for cosine similarity calculation [cite: 77, 80, 86]
        all_g.append(F.normalize(model.forward_graph(graphs), p=2, dim=-1))
        all_t.append(F.normalize(model.forward_text(text_emb), p=2, dim=-1))
    
    all_g = torch.cat(all_g, 0)
    all_t = torch.cat(all_t, 0)
    
    # Compute similarity matrix (Text x Graph)
    sims = all_t @ all_g.t()
    targets = torch.arange(all_t.size(0), device=device)
    
    # Calculate Mean Reciprocal Rank
    ranks = (sims.argsort(dim=-1, descending=True) == targets.view(-1, 1)).nonzero()[:, 1]
    mrr = (1.0 / (ranks.float() + 1.0)).mean().item()
    return mrr

# =========================================================
# 3. MAIN TRAINING LOOP
# =========================================================
def train_step_2(model, train_loader, val_loader, optimizer, scheduler, device, epochs):
    best_mrr = 0.0
    
    for ep in range(epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}")
        for graphs, text_emb in pbar:
            graphs, text_emb = graphs.to(device), text_emb.to(device)
            
            # Forward pass: Get normalized latent vectors [cite: 72, 73]
            g_vec, t_vec = model(graphs, text_emb)
            
            # Symmetric Contrastive Loss [cite: 74, 75]
            # Includes hard negatives in the logits matrix
            logits = (g_vec @ t_vec.T) / 0.07 
            labels = torch.arange(logits.size(0)).to(device)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # --- PER-EPOCH VALIDATION ---
        mrr = evaluate_retrieval(model, val_loader, device)
        print(f"Epoch {ep+1} Result | Avg Loss: {epoch_loss/len(train_loader):.4f} | Val MRR: {mrr:.4f}")
        
        # Save the best model based on validation performance [cite: 21, 29]
        if mrr > best_mrr:
            best_mrr = mrr
            torch.save(model.state_dict(), "models/model_strategy_8.pt")
            print(f"  >>> New Best MRR: {best_mrr:.4f}! Model Saved.")
        
        scheduler.step()

# =========================================================
# EXECUTION
# =========================================================
if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load SciBERT embeddings and PyG graphs [cite: 31, 32]
    train_emb = load_id2emb("embeddings/train_chembed_embeddings.csv")
    val_emb = load_id2emb("embeddings/validation_chembed_embeddings.csv")
    
    train_ds = PreprocessedGraphDataset("data/train_graphs.pkl", train_emb)
    val_ds = PreprocessedGraphDataset("data/validation_graphs.pkl", val_emb)

    # Initialize Hard Negative Sampler using results from Step 1
    sampler = HardNegativeSampler(train_ds, batch_size=24, hard_neg_path="strategy/strategy_8/hard_negatives.npy")
    train_loader = PyGDataLoader(train_ds, batch_sampler=sampler, collate_fn=collate_fn)
    val_loader = PyGDataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Model Setup (using your MolTransformerDual from previous prompt)
    model = MolTransformerDual(hidden=128, text_dim=768, out_dim=768).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

    train_step_2(model, train_loader, val_loader, optimizer, scheduler, DEVICE, epochs=100)

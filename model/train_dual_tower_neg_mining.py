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

from data_utils import (
    load_id2emb, load_descriptions_from_graphs,
    PreprocessedGraphDataset, collate_fn,
    x_map, e_map
)

# =========================================================
# CONFIGURATION
# =========================================================
TRAIN_GRAPHS = '/kaggle/input/molecular-data/train_graphs.pkl'
VAL_GRAPHS   = '/kaggle/input/molecular-data/validation_graphs.pkl' 
TEST_GRAPHS  = '/kaggle/input/molecular-data/test_graphs.pkl' 

# Using your specific embeddings
TRAIN_EMB_CSV = "/kaggle/working/ALTEGRAD-2025/train_stacked_embeddings.csv" 
VAL_EMB_CSV   = "/kaggle/working/ALTEGRAD-2025/validation_stacked_embeddings.csv"

# Output Paths
MODEL_PATH = "dual_tower_hard_neg.pt"

# Training Settings
# Set TRAIN_FULL_DATA = True for your FINAL run (uses Train + Val)
# Set TRAIN_FULL_DATA = False to monitor validation score first
TRAIN_FULL_DATA = True 

BATCH_SIZE = 24       
EPOCHS = 25           
LR = 2e-4             
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hard Negative Mining Settings
USE_HARD_NEGATIVES = True   # Enable hard negative mining
NUM_HARD_NEGATIVES = 10      # Number of hard negatives per sample
TEMPERATURE = 0.07          # Temperature for contrastive loss

# Cross-Modal Attention Settings
USE_CROSS_ATTENTION = True  # Enable cross-modal attention
CROSS_ATTN_HEADS = 8        # Number of attention heads
CROSS_ATTN_DROPOUT = 0.1    # Dropout for cross-attention

# =========================================================
# THE DUAL TOWER MODEL
# =========================================================

class LearnableTemperature(nn.Module):
    """
    Learnable temperature parameter for contrastive learning.
    Initialized at 0.07 and learned during training.
    Uses log-space to ensure temperature stays positive.
    """
    def __init__(self, init_temp=TEMPERATURE):
        super().__init__()
        # Store log(temperature) to ensure it stays positive
        self.log_temp = nn.Parameter(torch.log(torch.tensor(init_temp)))
    
    def forward(self):
        # Return exp(log_temp) to get actual temperature
        return torch.exp(self.log_temp)
    
    def get_temperature(self):
        """Get current temperature value"""
        return self.forward().item()


class MolTransformerDual(nn.Module):
    def __init__(self, hidden=128, text_dim=768, out_dim=768, layers=3, heads=4, use_cross_attn=True):
        super().__init__()
        
        self.use_cross_attn = use_cross_attn
        self.out_dim = out_dim
        
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
        
        # --- CROSS-MODAL ATTENTION ---
        if self.use_cross_attn:
            # Graph attends to Text
            self.cross_attn_g2t = nn.MultiheadAttention(
                embed_dim=out_dim,
                num_heads=CROSS_ATTN_HEADS,
                dropout=CROSS_ATTN_DROPOUT,
                batch_first=True
            )
            
            # Text attends to Graph
            self.cross_attn_t2g = nn.MultiheadAttention(
                embed_dim=out_dim,
                num_heads=CROSS_ATTN_HEADS,
                dropout=CROSS_ATTN_DROPOUT,
                batch_first=True
            )
            
            # Fusion layers after cross-attention
            self.graph_fusion = nn.Sequential(
                nn.Linear(out_dim * 2, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            
            self.text_fusion = nn.Sequential(
                nn.Linear(out_dim * 2, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        
        # --- LEARNABLE TEMPERATURE ---
        # Temperature parameter that will be optimized during training
        self.temperature = LearnableTemperature(init_temp=TEMPERATURE)

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
        """Training Step: Return both vectors with optional cross-attention"""
        # Extract features from both towers
        g_vec = self.forward_graph(batch)  # [B, D]
        t_vec = self.forward_text(text_emb)  # [B, D]
        
        if self.use_cross_attn:
            # Apply cross-modal attention
            # Add sequence dimension for attention: [B, D] -> [B, 1, D]
            g_vec_seq = g_vec.unsqueeze(1)
            t_vec_seq = t_vec.unsqueeze(1)
            
            # Graph attends to Text (query=graph, key=text, value=text)
            g_attended, _ = self.cross_attn_g2t(
                query=g_vec_seq,
                key=t_vec_seq,
                value=t_vec_seq
            )
            g_attended = g_attended.squeeze(1)  # [B, 1, D] -> [B, D]
            
            # Text attends to Graph (query=text, key=graph, value=graph)
            t_attended, _ = self.cross_attn_t2g(
                query=t_vec_seq,
                key=g_vec_seq,
                value=g_vec_seq
            )
            t_attended = t_attended.squeeze(1)  # [B, 1, D] -> [B, D]
            
            # Fusion: Concatenate original + attended, then project
            g_vec = self.graph_fusion(torch.cat([g_vec, g_attended], dim=-1))
            t_vec = self.text_fusion(torch.cat([t_vec, t_attended], dim=-1))
        
        # Normalize for contrastive learning
        g_vec = F.normalize(g_vec, p=2, dim=-1)
        t_vec = F.normalize(t_vec, p=2, dim=-1)
        
        return g_vec, t_vec

# =========================================================
# TRAINING UTILS
# =========================================================

@torch.no_grad()
def mine_hard_negatives(g_vec, t_vec, k=5):
    """
    Mine k hard negatives for each sample.
    Returns indices of the most similar (but incorrect) negatives.
    
    Args:
        g_vec: Graph embeddings [B, D]
        t_vec: Text embeddings [B, D]
        k: Number of hard negatives to mine per sample
    
    Returns:
        hard_neg_indices: [B, k] indices of hard negative texts for each graph
    """
    batch_size = g_vec.size(0)
    
    # Compute all pairwise similarities
    sim_matrix = g_vec @ t_vec.T  # [B, B]
    
    # Mask out the diagonal (correct pairs)
    mask = torch.eye(batch_size, device=sim_matrix.device, dtype=torch.bool)
    sim_matrix_masked = sim_matrix.masked_fill(mask, -1e9)
    
    # Get top-k most similar (hardest) negatives
    # These are the negatives that are most confusing for the model
    hard_neg_indices = sim_matrix_masked.topk(k, dim=1).indices  # [B, k]
    
    return hard_neg_indices


def contrastive_loss_with_hard_negatives(g_vec, t_vec, temperature=0.07, k_hard=5):
    """
    Contrastive loss with hard negative mining.
    
    For each sample, we consider:
    - 1 positive pair (correct match)
    - k hard negatives (most similar incorrect matches)
    - All other in-batch negatives
    
    Args:
        g_vec: Graph embeddings [B, D]
        t_vec: Text embeddings [B, D]
        temperature: Temperature for scaling
        k_hard: Number of hard negatives to emphasize
    """
    batch_size = g_vec.size(0)
    
    # Standard contrastive loss with all in-batch negatives
    logits = (g_vec @ t_vec.T) / temperature  # [B, B]
    labels = torch.arange(batch_size, device=g_vec.device)
    
    # Forward loss (graph -> text)
    loss_g2t = F.cross_entropy(logits, labels)
    
    # Backward loss (text -> graph)
    loss_t2g = F.cross_entropy(logits.T, labels)
    
    # Mine hard negatives
    hard_neg_idx = mine_hard_negatives(g_vec, t_vec, k=k_hard)
    
    # Additional loss term focusing on hard negatives
    # For each graph, create a focused loss with its positive and hard negatives
    hard_loss_g2t = 0.0
    hard_loss_t2g = 0.0
    
    for i in range(batch_size):
        # Get embeddings for sample i
        g_i = g_vec[i:i+1]  # [1, D]
        t_i = t_vec[i:i+1]  # [1, D]
        
        # Get hard negative texts for this graph
        hard_neg_texts = t_vec[hard_neg_idx[i]]  # [k, D]
        
        # Combine positive and hard negatives
        texts_combined = torch.cat([t_i, hard_neg_texts], dim=0)  # [k+1, D]
        
        # Compute logits
        logits_focused = (g_i @ texts_combined.T) / temperature  # [1, k+1]
        
        # Label is 0 (first position = positive)
        label_focused = torch.zeros(1, dtype=torch.long, device=g_vec.device)
        
        hard_loss_g2t += F.cross_entropy(logits_focused, label_focused)
        
        # Same for text -> graph direction
        hard_neg_graphs = g_vec[hard_neg_idx[i]]  # [k, D]
        graphs_combined = torch.cat([g_i, hard_neg_graphs], dim=0)  # [k+1, D]
        logits_focused_t2g = (t_i @ graphs_combined.T) / temperature
        hard_loss_t2g += F.cross_entropy(logits_focused_t2g, label_focused)
    
    # Average hard losses
    hard_loss_g2t /= batch_size
    hard_loss_t2g /= batch_size
    
    # Combine standard contrastive loss with hard negative loss
    # Weight hard negatives more heavily (0.5 weight)
    total_loss = (loss_g2t + loss_t2g) / 2 + 0.5 * (hard_loss_g2t + hard_loss_t2g) / 2
    
    return total_loss


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total = 0.0, 0
    
    for graphs, text_emb in loader:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)
        
        # 1. Forward both towers
        g_vec, t_vec = model(graphs, text_emb)
        
        # 2. Get learnable temperature
        temperature = model.temperature()
        
        # 3. Contrastive Loss with Hard Negative Mining
        if USE_HARD_NEGATIVES and g_vec.size(0) > NUM_HARD_NEGATIVES:
            loss = contrastive_loss_with_hard_negatives(
                g_vec, t_vec, 
                temperature=temperature, 
                k_hard=NUM_HARD_NEGATIVES
            )
        else:
            # Fallback to standard contrastive loss for small batches
            logits = (g_vec @ t_vec.T) / temperature
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

    # 3. Detect embedding dimension from first sample
    sample_emb = next(iter(train_emb.values()))
    text_embedding_dim = sample_emb.shape[0]
    print(f"Detected text embedding dimension: {text_embedding_dim}")
    
    # 4. Setup DataLoaders
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

    # 5. Model Setup
    # Use detected embedding dimension (768 for SciBERT, 3072 for stacked embeddings)
    model = MolTransformerDual(
        hidden=128, 
        text_dim=text_embedding_dim, 
        out_dim=768,
        use_cross_attn=USE_CROSS_ATTENTION
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    print(f"Cross-Modal Attention: {'Enabled' if USE_CROSS_ATTENTION else 'Disabled'}")
    print(f"Model text_dim: {text_embedding_dim} → out_dim: 768")
    
    # 6. Training Loop
    best_mrr = 0.0
    
    print(f"\n--- Starting Dual Tower Training ({EPOCHS} Epochs) ---")
    print(f"Initial Temperature: {model.temperature.get_temperature():.4f}")
    
    for ep in range(EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, DEVICE)
        current_temp = model.temperature.get_temperature()
        
        # Validation Logic
        if val_loader:
            val_scores = eval_retrieval(val_loader, model, DEVICE)
            print(f"Epoch {ep+1}/{EPOCHS} | Loss: {loss:.4f} | MRR: {val_scores.get('MRR', 0):.4f} | Temp: {current_temp:.4f}")
            
            # Save Best Model
            if val_scores.get('MRR', 0) > best_mrr:
                best_mrr = val_scores['MRR']
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"  >>> New Best Model Saved (MRR: {best_mrr:.4f})")
        else:
            # Blind Training (Full Data) - Just save the latest
            print(f"Epoch {ep+1}/{EPOCHS} | Loss: {loss:.4f} | Temp: {current_temp:.4f}")
            torch.save(model.state_dict(), MODEL_PATH)
            
        scheduler.step()
    
    print(f"\nFinal Temperature: {model.temperature.get_temperature():.4f}")

    print(f"\nTraining Complete. Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()

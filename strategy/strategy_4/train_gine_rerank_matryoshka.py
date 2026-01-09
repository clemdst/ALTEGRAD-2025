import os
import copy
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_geometric.data import Batch
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
TRAIN_GRAPHS = "./data/train_graphs.pkl"
VAL_GRAPHS   = "./data/validation_graphs.pkl"
TEST_GRAPHS  = "./data/test_graphs.pkl"

TRAIN_EMB_CSV = "embeddings/train_scibert_embeddings.csv"
VAL_EMB_CSV   = "embeddings/validation_scibert_embeddings.csv"

BATCH_SIZE = 32
EPOCHS = 20          
EPOCHS_RERANK = 10   
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_MODE = False
N_SAMPLES = 1000

TRAIN_RETRIEVER = False
RETRIEVER_PATH = "models/model_strategy_3.pt"

# --- NEW: MATRYOSHKA CONFIG ---
# We train the model to be accurate at ALL these dimensions simultaneously
MRL_DIMS = [64, 128, 256, 512, 768] 


# =========================================================
# MODEL 1: MATRYOSHKA RETRIEVER
# =========================================================
class MolGINE(nn.Module):
    def __init__(self, hidden=128, out_dim=768, layers=3):
        super().__init__()
        self.node_emb = nn.ModuleList([nn.Embedding(len(x_map[key]), hidden) for key in x_map])
        self.node_proj = nn.Linear(hidden * len(x_map), hidden)    
        self.edge_emb = nn.ModuleList([nn.Embedding(len(e_map[key]), hidden) for key in e_map])
        self.edge_proj = nn.Linear(hidden * len(e_map), hidden)
        
        self.convs = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(mlp))
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, batch: Batch):
        node_feats = [emb(batch.x[:, i]) for i, emb in enumerate(self.node_emb)]
        x = self.node_proj(torch.cat(node_feats, dim=-1))
        edge_feats = [emb(batch.edge_attr[:, i]) for i, emb in enumerate(self.edge_emb)]
        edge_feats = self.edge_proj(torch.cat(edge_feats, dim=-1))
        
        for conv in self.convs:
            x = F.relu(conv(x, batch.edge_index, edge_feats))
        g = global_add_pool(x, batch.batch)
        
        # --- CRITICAL CHANGE FOR MATRYOSHKA ---
        # Do NOT normalize here. We must slice first, then normalize in the loss function.
        return self.proj(g) 


# =========================================================
# MODEL 2: RESIDUAL RERANKER
# =========================================================
class MolGINE_Residual_Reranker(nn.Module):
    def __init__(self, hidden=128, text_dim=768, layers=3):
        super().__init__()
        
        # --- Graph Encoder (Same as Retriever) ---
        self.node_emb = nn.ModuleList([nn.Embedding(len(x_map[key]), hidden) for key in x_map])
        self.node_proj = nn.Linear(hidden * len(x_map), hidden)    
        self.edge_emb = nn.ModuleList([nn.Embedding(len(e_map[key]), hidden) for key in e_map])
        self.edge_proj = nn.Linear(hidden * len(e_map), hidden)
        
        self.convs = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(mlp))
            
        # 1. PROJECTION LAYER (Fix 1)
        self.proj = nn.Linear(hidden, text_dim)

        # --- The Correction Head ---
        # Input: [Graph_768; Text_768] -> 1536 dimensions
        self.cross_head = nn.Sequential(
            nn.Linear(text_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        nn.init.constant_(self.cross_head[-1].weight, 0.0)
        nn.init.constant_(self.cross_head[-1].bias, 0.0)

    def forward(self, batch: Batch, candidate_text_embs):
        # 1. Encode Graph
        node_feats = [emb(batch.x[:, i]) for i, emb in enumerate(self.node_emb)]
        x = self.node_proj(torch.cat(node_feats, dim=-1))
        edge_feats = [emb(batch.edge_attr[:, i]) for i, emb in enumerate(self.edge_emb)]
        edge_feats = self.edge_proj(torch.cat(edge_feats, dim=-1))
        
        for conv in self.convs:
            x = F.relu(conv(x, batch.edge_index, edge_feats))
        g_hidden = global_add_pool(x, batch.batch) 
        
        # 2. Project
        g_vec = self.proj(g_hidden) # [BS, 768]
        
        # 3. Expand
        k = candidate_text_embs.size(1)
        g_vec_expanded = g_vec.unsqueeze(1).expand(-1, k, -1)
        
        # 4. Base Score
        g_norm = F.normalize(g_vec_expanded, dim=-1)
        t_norm = F.normalize(candidate_text_embs, dim=-1)
        base_score = (g_norm * t_norm).sum(dim=-1, keepdim=True)
        
        # 5. Residual Correction
        joint_rep = torch.cat([g_vec_expanded, candidate_text_embs], dim=-1)
        bs, _, dim = joint_rep.size()
        delta = self.cross_head(joint_rep.view(bs * k, dim)).view(bs, k, 1)
        
        final_score = (base_score / 0.07) + delta
        
        # 2. SQUEEZE OUTPUT (Fix 2)
        return final_score.squeeze(-1) # [BS, K, 1] -> [BS, K]

    def load_from_retriever(self, retriever_model):
        print("Transferring weights from Retriever...")
        self.node_emb.load_state_dict(retriever_model.node_emb.state_dict())
        self.node_proj.load_state_dict(retriever_model.node_proj.state_dict())
        self.edge_emb.load_state_dict(retriever_model.edge_emb.state_dict())
        self.edge_proj.load_state_dict(retriever_model.edge_proj.state_dict())
        self.convs.load_state_dict(retriever_model.convs.state_dict())
        self.proj.load_state_dict(retriever_model.proj.state_dict())
# =========================================================
# DATA HELPERS
# =========================================================
class LimitedGraphDataset:
    def __init__(self, dataset, n_samples=None):
        self.dataset = dataset
        self.n_samples = min(n_samples, len(dataset)) if n_samples else len(dataset)
    def __len__(self): return self.n_samples
    def __getitem__(self, idx): 
        if idx >= self.n_samples: raise IndexError
        return self.dataset[idx]

# =========================================================
# TRAINING FUNCTIONS
# =========================================================

def train_epoch_matryoshka(mol_enc, loader, optimizer, device):
    """
    TRAINS WITH MATRYOSHKA LOSS:
    Calculates loss at [64, 128, 256, 512, 768] and sums them up.
    """
    mol_enc.train()
    total_loss, total = 0.0, 0
    
    for graphs, text_emb in loader:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device) # [BS, 768]
        
        # Get RAW output (Un-normalized)
        mol_raw = mol_enc(graphs) # [BS, 768]
        
        batch_loss = 0.0
        
        # --- MATRYOSHKA LOOP ---
        for dim in MRL_DIMS:
            if dim > mol_raw.size(1): break
            
            # 1. Slice
            mol_slice = mol_raw[:, :dim]
            txt_slice = text_emb[:, :dim]
            
            # 2. Normalize
            mol_norm = F.normalize(mol_slice, p=2, dim=-1)
            txt_norm = F.normalize(txt_slice, p=2, dim=-1)
            
            # 3. Contrastive Loss (CLIP-style)
            logits = mol_norm @ txt_norm.T / 0.07
            labels = torch.arange(logits.size(0)).to(device)
            
            loss_i = F.cross_entropy(logits, labels)
            loss_t = F.cross_entropy(logits.T, labels)
            batch_loss += (loss_i + loss_t) / 2

        # Average loss across dimensions to keep scale consistent
        final_loss = batch_loss / len(MRL_DIMS)

        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()
        
        total_loss += final_loss.item() * graphs.num_graphs
        total += graphs.num_graphs
        
    return total_loss / total

def eval_matryoshka(loader, mol_enc, device):
    """Evaluates MRR at the 'Sweet Spot' dimension (e.g., 256 or Max)"""
    mol_enc.eval()
    all_mol_raw, all_txt_raw = [], []
    
    with torch.no_grad():
        for graphs, text_emb in loader:
            graphs = graphs.to(device)
            text_emb = text_emb.to(device)
            all_mol_raw.append(mol_enc(graphs))
            all_txt_raw.append(text_emb)
            
    if not all_mol_raw: return {}
    all_mol_raw = torch.cat(all_mol_raw, 0)
    all_txt_raw = torch.cat(all_txt_raw, 0)
    
    results = {}
    
    # Check specific dim (e.g. 256) performance
    eval_dim = 256 
    
    # Slice & Normalize
    mol_vec = F.normalize(all_mol_raw[:, :eval_dim], dim=-1)
    txt_vec = F.normalize(all_txt_raw[:, :eval_dim], dim=-1)
    
    sims = txt_vec @ mol_vec.t()
    ranks = sims.argsort(dim=-1, descending=True)
    correct = torch.arange(all_txt_raw.size(0), device=sims.device)
    pos = (ranks == correct.unsqueeze(1)).nonzero()[:, 1] + 1
    mrr = (1.0 / pos.float()).mean().item()
    
    return {"MRR@256": mrr}

def train_epoch_reranker(retriever, reranker, loader, full_text_matrix, optimizer, device, k=20):
    retriever.eval()
    reranker.train()
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()
    
    if full_text_matrix.device != device:
        full_text_matrix = full_text_matrix.to(device)

    # Use 256 dim for the Hard Negative Mining (It's the sweet spot)
    mining_dim = 256

    for graphs, text_emb in loader:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)
        bs = text_emb.size(0)
        
        # 1. Mine Negatives using Retriever (at 256 dim)
        with torch.no_grad():
            # Get raw output
            mol_raw = retriever(graphs)
            
            # Slice and Normalize for Mining
            mol_vec = F.normalize(mol_raw[:, :mining_dim], dim=-1)
            
            # Also slice global matrix for mining
            # Assuming full_text_matrix is already normalized 768-dim, we slice it
            # Note: Re-normalizing after slice is safer
            global_texts_slice = F.normalize(full_text_matrix[:, :mining_dim], dim=-1)
            
            sim_matrix = mol_vec @ global_texts_slice.t()
            _, topk_indices = sim_matrix.topk(k, dim=1)
            topk_indices = topk_indices.cpu()

        # 2. Prepare Reranker Batch
        candidate_list, label_list = [], []
        
        for i in range(bs):
            indices = topk_indices[i].numpy()
            
            # Get FULL dimension candidates for the reranker (Reranker can use 768)
            cands = full_text_matrix[indices] 
            labels = torch.zeros(k).to(device)
            
            # Teacher Forcing
            cands[-1] = text_emb[i] # Ground truth (Full dim)
            labels[-1] = 1.0
            
            candidate_list.append(cands)
            label_list.append(labels)
            
        candidates = torch.stack(candidate_list).to(device)
        targets = torch.stack(label_list).to(device)
        
        # 3. Train
        scores = reranker(graphs, candidates)
        loss = criterion(scores, targets)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def eval_hybrid(loader, retriever, reranker, device, k=20):
    retriever.eval()
    reranker.eval()
    
    all_txt_embs = []
    for _, text_emb in loader:
        all_txt_embs.append(text_emb.to(device)) # Keep raw for now
    all_txt_pool = torch.cat(all_txt_embs, dim=0)
    
    # Mining Dim
    mining_dim = 256
    # Slice pool for retrieval
    pool_mining = F.normalize(all_txt_pool[:, :mining_dim], dim=-1)
    
    rr_sum = 0.0
    total_count = 0
    global_idx = 0
    
    for graphs, _ in loader:
        graphs = graphs.to(device)
        bs = graphs.num_graphs
        
        # Stage 1: Retrieve (256 dim)
        mol_raw = retriever(graphs)
        mol_vec = F.normalize(mol_raw[:, :mining_dim], dim=-1)
        
        sim_matrix = mol_vec @ pool_mining.t()
        _, topk_indices = sim_matrix.topk(min(k, pool_mining.size(0)), dim=1)
        
        # Stage 2: Rerank (Full Dim)
        # Use full dimension embeddings for the Neural Reranker
        candidates = all_txt_pool[topk_indices] 
        
        rerank_scores = reranker(graphs, candidates)
        reranked_order = rerank_scores.argsort(dim=1, descending=True)
        
        topk_indices = topk_indices.cpu().numpy()
        reranked_order = reranked_order.cpu().numpy()
        
        for i in range(bs):
            true_idx = global_idx + i
            if true_idx in topk_indices[i]:
                cand_pos = (topk_indices[i] == true_idx).nonzero()[0][0]
                rank = (reranked_order[i] == cand_pos).nonzero()[0][0] + 1
                rr_sum += 1.0 / rank
            else:
                rr_sum += 0.0
                
        total_count += bs
        global_idx += bs
        
    return {"Hybrid_MRR": rr_sum / total_count if total_count > 0 else 0}

# =========================================================
# MAIN
# =========================================================
def main():
    print(f"Device: {DEVICE}")

    train_emb = load_id2emb(TRAIN_EMB_CSV)
    val_emb = load_id2emb(VAL_EMB_CSV) if os.path.exists(VAL_EMB_CSV) else None
    emb_dim = len(next(iter(train_emb.values()))) # Should be 768
    
    full_train_ds = PreprocessedGraphDataset(TRAIN_GRAPHS, train_emb)
    train_ds = LimitedGraphDataset(full_train_ds, N_SAMPLES) if TEST_MODE else full_train_ds
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    val_dl = None
    if val_emb and os.path.exists(VAL_GRAPHS):
        full_val_ds = PreprocessedGraphDataset(VAL_GRAPHS, val_emb)
        val_ds = LimitedGraphDataset(full_val_ds, N_SAMPLES) if TEST_MODE else full_val_ds
        val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # --- PREPARE GLOBAL TEXT MATRIX ---
    print("Preparing Global Text Matrix...")
    all_train_text_list = []
    for i in range(len(train_ds)):
        _, txt = train_ds[i]
        all_train_text_list.append(txt)
    full_train_text_matrix = torch.stack(all_train_text_list).to(DEVICE)
    # Don't normalize yet, we slice and normalize inside the mining function
    
    # --- STAGE 1: MATRYOSHKA RETRIEVER ---
    mol_enc = MolGINE(out_dim=emb_dim).to(DEVICE)
    
    if TRAIN_RETRIEVER:
        print("\n=== STAGE 1: Training Matryoshka Retriever ===")
        opt = torch.optim.Adam(mol_enc.parameters(), lr=LR)
        best_mrr = 0
        best_w = None
        
        for ep in range(EPOCHS):
            # Using the new Matryoshka training function
            loss = train_epoch_matryoshka(mol_enc, train_dl, opt, DEVICE)
            val_scores = eval_matryoshka(val_dl, mol_enc, DEVICE) if val_dl else {}
            print(f"S1 Epoch {ep+1}/{EPOCHS} - loss={loss:.4f} - {val_scores}")
            
            if val_scores.get("MRR@256", 0) > best_mrr:
                best_mrr = val_scores["MRR@256"]
                best_w = copy.deepcopy(mol_enc.state_dict())
                torch.save(mol_enc.state_dict(), RETRIEVER_PATH)
                
        if best_w: mol_enc.load_state_dict(best_w)
    else:
        print(f"\n=== STAGE 1: Loading Retriever from {RETRIEVER_PATH} ===")
        mol_enc.load_state_dict(torch.load(RETRIEVER_PATH, map_location=DEVICE))

    # --- STAGE 2: RESIDUAL RERANKER ---
    print("\n=== STAGE 2: Training Residual Reranker ===")
    
    reranker = MolGINE_Residual_Reranker(hidden=128, text_dim=emb_dim).to(DEVICE)
    reranker.load_from_retriever(mol_enc)
    
    # Freeze GNN initially
    print("Freezing Reranker GNN layers...")
    for name, param in reranker.named_parameters():
        if "cross_head" not in name:
            param.requires_grad = False
    
    reranker_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, reranker.parameters()), lr=1e-3)
    best_rerank_mrr = 0.0
    
    for ep in range(EPOCHS_RERANK):
        if ep == 2:
            print(">>> Unfreezing Reranker GNN...")
            for param in reranker.parameters():
                param.requires_grad = True
            reranker_opt = torch.optim.Adam(reranker.parameters(), lr=1e-4)

            # CRITICAL FIX: Use very low LR for the GNN, higher LR for the Head
            reranker_opt = torch.optim.Adam([
                {'params': reranker.convs.parameters(), 'lr': 1e-5},      # Very Slow (Don't break GNN)
                {'params': reranker.node_emb.parameters(), 'lr': 1e-5},
                {'params': reranker.edge_emb.parameters(), 'lr': 1e-5},
                {'params': reranker.cross_head.parameters(), 'lr': 1e-4}  # Normal Speed
            ])

        loss = train_epoch_reranker(mol_enc, reranker, train_dl, full_train_text_matrix, reranker_opt, DEVICE, k=20)
        
        val_scores = {}
        if val_dl:
            val_scores = eval_hybrid(val_dl, mol_enc, reranker, DEVICE, k=20)
        
        print(f"S2 Epoch {ep+1}/{EPOCHS_RERANK} - loss={loss:.4f} - {val_scores}")
        
        if val_scores.get("Hybrid_MRR", 0) > best_rerank_mrr:
            best_rerank_mrr = val_scores["Hybrid_MRR"]
            torch.save(reranker.state_dict(), "models/model_strategy_4.pt")

if __name__ == "__main__":
    main()

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
TRAIN_GRAPHS = "/kaggle/input/data-challenge-altegrad/train_graphs.pkl"
VAL_GRAPHS   = "/kaggle/input/data-challenge-altegrad/validation_graphs.pkl"
TEST_GRAPHS  = "/kaggle/input/data-challenge-altegrad/test_graphs.pkl"

TRAIN_EMB_CSV = "/kaggle/working/advanced_project_clone-/train_embeddings.csv"
VAL_EMB_CSV   = "/kaggle/working/advanced_project_clone-/validation_embeddings.csv"

BATCH_SIZE = 32
EPOCHS = 50
EPOCHS_rerank= 10
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_MODE = True
N_SAMPLES = 10

# Logic Flags
TRAIN_RETRIEVER = False  # Set False to load existing .pt file
RETRIEVER_PATH = "/kaggle/working/advanced_project_clone-/model_checkpoint.pt"


# =========================================================
# MODELS
# =========================================================
class MolGINE(nn.Module):
    def __init__(self, hidden=128, out_dim=256, layers=3):
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
        return F.normalize(self.proj(g), dim=-1)

class MolGINE_Residual_Reranker(nn.Module):
    def __init__(self, hidden=128, text_dim=768, layers=3):
        super().__init__()
        
        # --- Graph Encoder (Same as before) ---
        self.node_emb = nn.ModuleList([nn.Embedding(len(x_map[key]), hidden) for key in x_map])
        self.node_proj = nn.Linear(hidden * len(x_map), hidden)    
        self.edge_emb = nn.ModuleList([nn.Embedding(len(e_map[key]), hidden) for key in e_map])
        self.edge_proj = nn.Linear(hidden * len(e_map), hidden)
        
        self.convs = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(mlp))
            
        # --- The Correction Head ---
        # We process the interaction to find a "delta" to add to the dot product
        self.cross_head = nn.Sequential(
            nn.Linear(hidden + text_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Output delta
        )
        
        # Initialize the last layer to near-zero
        # This ensures the model starts exactly as the Retriever
        nn.init.constant_(self.cross_head[-1].weight, 0.0)
        nn.init.constant_(self.cross_head[-1].bias, 0.0)

    def forward(self, batch: Batch, candidate_text_embs):
        # 1. Encode Graph [BS, Hidden]
        node_feats = [emb(batch.x[:, i]) for i, emb in enumerate(self.node_emb)]
        x = self.node_proj(torch.cat(node_feats, dim=-1))
        edge_feats = [emb(batch.edge_attr[:, i]) for i, emb in enumerate(self.edge_emb)]
        edge_feats = self.edge_proj(torch.cat(edge_feats, dim=-1))
        
        for conv in self.convs:
            x = F.relu(conv(x, batch.edge_index, edge_feats))
        g_vec = global_add_pool(x, batch.batch)
        
        # 2. Expand Graph Vector [BS, K, Hidden]
        k = candidate_text_embs.size(1)
        g_vec_expanded = g_vec.unsqueeze(1).expand(-1, k, -1)
        
        # --- CRITICAL CHANGE: Calculate Base Dot Product ---
        # We normalize locally to match the Retriever's behavior
        g_norm = F.normalize(g_vec_expanded, dim=-1)
        t_norm = F.normalize(candidate_text_embs, dim=-1)
        
        # The base score from the Retriever (Cosine Similarity)
        # element-wise multiply and sum over dim -1
        base_score = (g_norm * t_norm).sum(dim=-1, keepdim=True) # [BS, K, 1]
        
        # 3. Calculate Correction (Delta)
        # We feed un-normalized vectors to MLP for richer features
        joint_rep = torch.cat([g_vec_expanded, candidate_text_embs], dim=-1)
        
        bs, _, dim = joint_rep.size()
        delta = self.cross_head(joint_rep.view(bs * k, dim)).view(bs, k, 1)
        
        # 4. Final Score = Base + Delta
        # We scale base_score by a factor (e.g. 10) because dot products are small (~0.3) 
        # while logits are usually large (~2.0). 
        # Or we let the MLP learn the scale. 
        # Simple addition usually works best if we trained Retriever with temperature.
        
        final_logits = (base_score / 0.07) + delta
        
        return final_logits.squeeze(-1) # [BS, K]

    def load_from_retriever(self, retriever_model):
        print("Transferring weights and Freezing GNN initially...")
        self.node_emb.load_state_dict(retriever_model.node_emb.state_dict())
        self.node_proj.load_state_dict(retriever_model.node_proj.state_dict())
        self.edge_emb.load_state_dict(retriever_model.edge_emb.state_dict())
        self.edge_proj.load_state_dict(retriever_model.edge_proj.state_dict())
        self.convs.load_state_dict(retriever_model.convs.state_dict())
# =========================================================
# HELPERS
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
# TRAIN & EVAL FUNCTIONS
# =========================================================
def train_epoch_retriever(mol_enc, loader, optimizer, device):
    mol_enc.train()
    total_loss, total = 0.0, 0
    for graphs, text_emb in loader:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)
        mol_vec = mol_enc(graphs)
        txt_vec = F.normalize(text_emb, dim=-1)
        logits = mol_vec @ txt_vec.T / 0.07
        labels = torch.arange(logits.size(0)).to(device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item() * graphs.num_graphs; total += graphs.num_graphs
    return total_loss / total

def eval_retriever(loader, mol_enc, device):
    mol_enc.eval()
    all_mol, all_txt = [], []
    with torch.no_grad():
        for graphs, text_emb in loader:
            graphs = graphs.to(device)
            text_emb = text_emb.to(device)
            all_mol.append(mol_enc(graphs))
            all_txt.append(F.normalize(text_emb, dim=-1))
    if not all_mol: return {}
    all_mol = torch.cat(all_mol, 0); all_txt = torch.cat(all_txt, 0)
    sims = all_txt @ all_mol.t()
    ranks = sims.argsort(dim=-1, descending=True)
    correct = torch.arange(all_txt.size(0), device=sims.device)
    pos = (ranks == correct.unsqueeze(1)).nonzero()[:, 1] + 1
    mrr = (1.0 / pos.float()).mean().item()
    return {"MRR": mrr}

def train_epoch_reranker(retriever, reranker, loader, optimizer, device, k=10):
    retriever.eval()
    reranker.train()
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()
    
    for graphs, text_emb in loader:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)
        bs = text_emb.size(0)
        
        # 1. Mine Negatives with Retriever
        with torch.no_grad():
            mol_vec = retriever(graphs)
            sim_matrix = mol_vec @ text_emb.t()
            _, topk_indices = sim_matrix.topk(min(k, bs), dim=1)
            topk_indices = topk_indices.cpu()

        # 2. Prepare Reranker Batch
        candidate_list, label_list = [], []
        for i in range(bs):
            indices = topk_indices[i].numpy()
            if i not in indices: indices[-1] = i # Teacher Forcing
            
            candidate_list.append(text_emb[indices])
            label_list.append(torch.tensor([1.0 if idx == i else 0.0 for idx in indices]))
            
        candidates = torch.stack(candidate_list).to(device)
        targets = torch.stack(label_list).to(device)
        
        # 3. Train
        scores = reranker(graphs, candidates)
        loss = criterion(scores, targets)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def eval_two_stage(loader, retriever, reranker, device, k=5):
    """
    Validation for Reranker:
    1. Retrieve Top-K using Retriever (Stage 1).
    2. Rerank these K using Reranker (Stage 2).
    3. Compute MRR based on final order.
    """
    retriever.eval()
    reranker.eval()
    
    # 1. Collect ALL Validation Text Embeddings first (to serve as the retrieval pool)
    # Note: In this dataset, graph_i corresponds to text_i. 
    # We need the full pool to calculate retrieval ranks correctly.
    all_txt_embs = []
    for _, text_emb in loader:
        all_txt_embs.append(F.normalize(text_emb.to(device), dim=-1))
    all_txt_pool = torch.cat(all_txt_embs, dim=0) # [Total_Val, Dim]
    
    rr_sum = 0.0
    total_count = 0
    
    # 2. Iterate Graphs and Rerank
    # We iterate loader again to get graphs (Batch objects)
    global_idx = 0
    
    for graphs, text_emb in loader:
        graphs = graphs.to(device)
        bs = graphs.num_graphs
        
        # --- STAGE 1: Retrieval ---
        mol_vec = retriever(graphs)
        # Compare batch graphs vs ALL validation texts
        sim_matrix = mol_vec @ all_txt_pool.t() # [BS, Total_Val]
        
        # Get Top K candidates
        # We need slightly more than K if we want to check if truth is in top K?
        # Standard: Take top K. If truth not in K, reciprocal rank is based on Stage 1 (or 0).
        # Let's assume we refine the Top K.
        curr_k = min(k, all_txt_pool.size(0))
        _, topk_indices = sim_matrix.topk(curr_k, dim=1) # [BS, K]
        
        # --- STAGE 2: Reranking ---
        # Fetch the embeddings for the top K candidates
        candidates = all_txt_pool[topk_indices] # [BS, K, Dim]
        
        # Run Reranker
        rerank_scores = reranker(graphs, candidates) # [BS, K]
        rerank_scores = torch.sigmoid(rerank_scores)
        
        # Sort the K candidates by Reranker score
        # argsort descending
        reranked_order = rerank_scores.argsort(dim=1, descending=True) # [BS, K]
        
        # --- Calculate Metrics ---
        topk_indices = topk_indices.cpu().numpy()
        reranked_order = reranked_order.cpu().numpy()
        
        for i in range(bs):
            true_idx = global_idx + i
            
            # Find if true_idx is in the Top K retrieved
            # topk_indices[i] contains the global indices of texts
            retrieved_ids = topk_indices[i]
            
            if true_idx in retrieved_ids:
                # It was retrieved! Where did the Reranker put it?
                # Find the position of true_idx in the 'retrieved_ids' array
                # e.g. retrieved_ids = [50, 100, 7, ...] and true_idx is 7. index is 2.
                candidate_pos = (retrieved_ids == true_idx).nonzero()[0][0]
                
                # Now where is 'candidate_pos' in the 'reranked_order'?
                # reranked_order[i] is e.g. [2, 0, 1, 3...] meaning candidate 2 is ranked 1st.
                rank_in_rerank = (reranked_order[i] == candidate_pos).nonzero()[0][0]
                
                # Rank is 1-based
                final_rank = rank_in_rerank + 1
                rr_sum += 1.0 / final_rank
            else:
                # Not retrieved in Top K. 
                # Strict Reranker MRR treats this as 0 (failed to recall).
                # Hybrid MRR would use the retriever's rank (likely > K).
                # For "Validation Score" of Reranker, 0 is standard if we only look at Top K.
                rr_sum += 0.0
                
        total_count += bs
        global_idx += bs
        
    return {"MRR": rr_sum / total_count if total_count > 0 else 0}

# =========================================================
# MAIN
# =========================================================
def main():
    print(f"Device: {DEVICE}")

    # Load Maps
    train_emb = load_id2emb(TRAIN_EMB_CSV)
    val_emb = load_id2emb(VAL_EMB_CSV) if os.path.exists(VAL_EMB_CSV) else None
    emb_dim = len(next(iter(train_emb.values())))
    
    # Load Data
    full_train_ds = PreprocessedGraphDataset(TRAIN_GRAPHS, train_emb)
    train_ds = LimitedGraphDataset(full_train_ds, N_SAMPLES) if TEST_MODE else full_train_ds
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    # Validation Data
    val_dl = None
    if val_emb and os.path.exists(VAL_GRAPHS):
        full_val_ds = PreprocessedGraphDataset(VAL_GRAPHS, val_emb)
        val_ds = LimitedGraphDataset(full_val_ds, N_SAMPLES) if TEST_MODE else full_val_ds
        val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn) # Batch 32 for eval safety

    # --- STAGE 1: RETRIEVER ---
    mol_enc = MolGINE(out_dim=emb_dim).to(DEVICE)
    
    if TRAIN_RETRIEVER:
        print("\n=== STAGE 1: Training Retriever ===")
        opt = torch.optim.Adam(mol_enc.parameters(), lr=LR)
        best_mrr = 0
        best_w = None
        
        for ep in range(EPOCHS):
            loss = train_epoch_retriever(mol_enc, train_dl, opt, DEVICE)
            val_scores = eval_retriever(val_dl, mol_enc, DEVICE) if val_dl else {}
            print(f"S1 Epoch {ep+1}/{EPOCHS} - loss={loss:.4f} - val={val_scores}")
            
            if val_scores.get("MRR", 0) > best_mrr:
                best_mrr = val_scores["MRR"]
                best_w = copy.deepcopy(mol_enc.state_dict())
                
        if best_w: mol_enc.load_state_dict(best_w)
        torch.save(mol_enc.state_dict(), RETRIEVER_PATH)
    else:
        print(f"\n=== STAGE 1: Loading Retriever from {RETRIEVER_PATH} ===")
        mol_enc.load_state_dict(torch.load(RETRIEVER_PATH, map_location=DEVICE))

    # # --- STAGE 2: RERANKER ---
    # print("\n=== STAGE 2: Training Reranker ===")
    # mol_enc.eval() 
    # for p in mol_enc.parameters(): p.requires_grad = False # Freeze Stage 1
    
    # reranker = MolGINE_Reranker(text_dim=emb_dim).to(DEVICE)
    # reranker.load_from_retriever(mol_enc)
    # reranker_opt = torch.optim.Adam(reranker.parameters(), lr=1e-4)
    
    # # Eval Baseline first
    # if val_dl:
    #     base_scores = eval_two_stage(val_dl, mol_enc, reranker, DEVICE, k=10)
    #     print(f"Initial Baseline (Untrained Reranker) - val={base_scores}")

    # best_rerank_mrr = 0.0
    
    # for ep in range(EPOCHS_rerank): # Reranker Epochs
    #     # Train
    #     loss = train_epoch_reranker(mol_enc, reranker, train_dl, reranker_opt, DEVICE, k=10)
        
    #     # Validation
    #     val_scores = {}
    #     if val_dl:
    #         val_scores = eval_two_stage(val_dl, mol_enc, reranker, DEVICE, k=10)
        
    #     print(f"S2 Epoch {ep+1}/{EPOCHS_rerank} - loss={loss:.4f} - val={val_scores}")
        
    #     if val_scores.get("MRR", 0) > best_rerank_mrr:
    #         best_rerank_mrr = val_scores["MRR"]
    #         torch.save(reranker.state_dict(), "reranker_best.pt")
    print("\n=== STAGE 2: Training Residual Reranker ===")
    
    # 1. Initialize Residual Reranker
    reranker = MolGINE_Residual_Reranker(hidden=128, text_dim=emb_dim).to(DEVICE)
    reranker.load_from_retriever(mol_enc)
    
    # 2. FREEZE STRATEGY
    # Freeze GNN layers in the Reranker initially
    print("Freezing Reranker GNN layers (Training Head Only)...")
    for name, param in reranker.named_parameters():
        if "cross_head" not in name:
            param.requires_grad = False
    
    # Optimizer (Model 1 is frozen, Model 2 GNN is frozen, Model 2 Head is trainable)
    # We filter parameters to only pass trainable ones to Adam
    reranker_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, reranker.parameters()), lr=1e-3)

    best_rerank_mrr = 0.0
    
    # Epochs
    for ep in range(10):
        # UNFREEZE after Epoch 1
        if ep == 1:
            print("Unfreezing Reranker GNN (Fine-tuning all layers)...")
            for param in reranker.parameters():
                param.requires_grad = True
            # Re-create optimizer with lower LR for fine-tuning
            reranker_opt = torch.optim.Adam(reranker.parameters(), lr=1e-4)

        loss = train_epoch_reranker(mol_enc, reranker, train_dl, reranker_opt, DEVICE, k=10)
        
        val_scores = {}
        if val_dl:
            # Use Hybrid Eval to see true performance
            val_scores = eval_hybrid(val_dl, mol_enc, reranker, DEVICE, k=10)
        
        print(f"S2 Epoch {ep+1}/10 - loss={loss:.4f} - {val_scores}")
        
        if val_scores.get("Hybrid_MRR", 0) > best_rerank_mrr:
            best_rerank_mrr = val_scores["Hybrid_MRR"]
            torch.save(reranker.state_dict(), "reranker_best.pt")

if __name__ == "__main__":
    main()

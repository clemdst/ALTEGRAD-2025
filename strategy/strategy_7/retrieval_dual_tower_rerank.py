
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data_utils import (
    load_id2emb, load_descriptions_from_graphs, 
    PreprocessedGraphDataset, collate_fn
)

# --- IMPORT MODEL CLASSES ---
# Ensure these files (train.py and train_reranker.py) are in the same directory
from strategy.strategy_5.train_dual_tower import MolTransformerDual  # Phase 6 Backbone
from train_dual_tower_reranker import RerankNet  # Phase 7 Reranker

# ==========================================
# CONFIGURATION
# ==========================================
# 1. Data Paths
TRAIN_GRAPHS = "data/train_graphs.pkl"
VAL_GRAPHS   = "data/validation_graphs.pkl"
TEST_GRAPHS  = "data/test_graphs.pkl"

# Using your specific SciBERT embeddings
TRAIN_EMB_CSV = "embeddings/train_chembed_embeddings.csv"
VAL_EMB_CSV   = "embeddings/validation_chembed_embeddings.csv"

# Models
BACKBONE_PATH = "models/model_strategy_6.pt"       # Phase 6 Model
RERANKER_PATH = "models/model_strategy_7.pt"       # Phase 7 Model

# 3. Settings
SUBMISSION_CSV = "results/strategy_7_test_descriptions.csv"
BATCH_SIZE = 32
TOP_K = 50       # Retrieve top 50 candidates, then let the Reranker pick the best one
HIDDEN_DIM = 128 # MUST match your saved checkpoint (128 based on previous error)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def generate_submission():
    print(f"--- Phase 7 Inference (Retrieve & Rerank) on {DEVICE} ---")
    
    # ==========================================
    # 1. LOAD MODELS
    # ==========================================
    print("1. Loading Models...")
    
    # A. Load Backbone (Dual Tower)
    print(f"   Loading Backbone from {BACKBONE_PATH}...")
    backbone = MolTransformerDual(hidden=HIDDEN_DIM, text_dim=768, out_dim=768).to(DEVICE)
    if not os.path.exists(BACKBONE_PATH):
        raise FileNotFoundError(f"Backbone not found at {BACKBONE_PATH}")
    
    # Load safely (map_location handles CPU/GPU mismatch)
    map_loc = "cpu" if DEVICE == "cpu" else None
    backbone.load_state_dict(torch.load(BACKBONE_PATH, map_location=map_loc))
    backbone.eval()
    
    # B. Load Reranker
    print(f"   Loading Reranker from {RERANKER_PATH}...")
    reranker = RerankNet(input_dim=768*2).to(DEVICE)
    if not os.path.exists(RERANKER_PATH):
        raise FileNotFoundError(f"Reranker not found at {RERANKER_PATH}")
    reranker.load_state_dict(torch.load(RERANKER_PATH, map_location=map_loc))
    reranker.eval()

    # ==========================================
    # 2. PREPARE TEXT LIBRARY
    # ==========================================
    print("2. Building Text Index...")
    
    # Load Raw Embeddings
    train_emb = load_id2emb(TRAIN_EMB_CSV)
    val_emb = load_id2emb(VAL_EMB_CSV)
    full_emb_dict = {**train_emb, **val_emb}
    
    # Load Raw Text (for final output)
    train_desc = load_descriptions_from_graphs(TRAIN_GRAPHS)
    val_desc = load_descriptions_from_graphs(VAL_GRAPHS)
    full_id2desc = {**train_desc, **val_desc}
    
    candidate_ids = list(full_emb_dict.keys())
    print(f"   Total Library Size: {len(candidate_ids)} captions")
    
    # --- LIBRARY UPGRADE (Backbone Projection) ---
    print("   Projecting Library to Shared Space...")
    all_raw_embs = torch.stack([full_emb_dict[id_] for id_ in candidate_ids])
    
    library_vecs = []
    chunk_size = 1024
    
    for i in range(0, len(all_raw_embs), chunk_size):
        batch_raw = all_raw_embs[i:i+chunk_size].to(DEVICE)
        
        # Project through Text Tower (Phase 6)
        batch_proj = backbone.forward_text(batch_raw)
        batch_norm = F.normalize(batch_proj, p=2, dim=-1)
        
        library_vecs.append(batch_norm)
        
    library_tensor = torch.cat(library_vecs, dim=0) # [N_Library, 768]
    print(f"   Library Ready. Shape: {library_tensor.size()}")

    # ==========================================
    # 3. PROCESS TEST SET (With Dummy Emb Fix)
    # ==========================================
    print("3. Processing Test Graphs...")
    
    # A. Get Test IDs first
    # Initialize with empty dict just to read the graph objects
    temp_dataset = PreprocessedGraphDataset(TEST_GRAPHS, {}) 
    
    # Robustly get graph list depending on dataset implementation
    if hasattr(temp_dataset, 'graphs'):
        test_graph_list = temp_dataset.graphs
    elif hasattr(temp_dataset, 'dataset'):
        test_graph_list = temp_dataset.dataset.graphs
    else:
        # Fallback assumption (list of Data objects)
        test_graph_list = temp_dataset 
        
    test_ids_list = [g.id for g in test_graph_list]
    
    # B. Create Dummy Embeddings (The Fix!)
    # The dataset needs *something* to return for text, even if we ignore it.
    print(f"   Generating dummy embeddings for {len(test_ids_list)} test molecules...")
    dummy_emb_dict = {
        g_id: torch.zeros(768) 
        for g_id in test_ids_list
    }
    
    # C. Initialize Real Dataset
    test_ds = PreprocessedGraphDataset(TEST_GRAPHS, dummy_emb_dict)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    results = []
    global_idx = 0
    
    for batch in tqdm(test_dl, desc="Inference"):
        # Unpack: graphs, dummy_text (we ignore dummy_text)
        if isinstance(batch, (tuple, list)):
            graphs, _ = batch 
        else:
            graphs = batch 
            
        graphs = graphs.to(DEVICE)
        batch_size_curr = graphs.num_graphs
        
        # --- A. Graph Encoding (Backbone) ---
        mol_vecs = backbone.forward_graph(graphs)
        mol_vecs = F.normalize(mol_vecs, p=2, dim=-1) # [Batch, 768]
        
        # --- B. Coarse Retrieval (Top-K) ---
        # 1. Compute Cosine Similarity
        sim_matrix = mol_vecs @ library_tensor.t()
        
        # 2. Retrieve Top-K candidates (e.g., Top 50)
        _, topk_indices = sim_matrix.topk(TOP_K, dim=1) # [Batch, K]
        
        # --- C. Fine-Grained Reranking ---
        topk_indices = topk_indices.cpu().numpy()
        
        for i in range(batch_size_curr):
            # 1. Prepare Inputs for Reranker
            # Repeat the single graph vector K times to match the K candidates
            # Shape: [1, 768] -> [K, 768]
            curr_mol_vec = mol_vecs[i].unsqueeze(0).repeat(TOP_K, 1) 
            
            # Fetch the K candidate vectors from the library
            cand_indices = topk_indices[i] 
            curr_cand_vecs = library_tensor[cand_indices] # [K, 768]
            
            # 2. Score with Reranker (Phase 7 Model)
            rerank_scores = reranker(curr_mol_vec, curr_cand_vecs) # [K, 1]
            rerank_scores = rerank_scores.squeeze() # [K]
            
            # 3. Pick the Winner (Argmax of Reranker Score)
            best_local_idx = torch.argmax(rerank_scores).item()
            winner_global_idx = cand_indices[best_local_idx]
            
            # 4. Map to Text
            pred_db_id = candidate_ids[winner_global_idx]
            pred_desc = full_id2desc[pred_db_id]
            current_test_id = test_ids_list[global_idx]
            
            results.append({
                'ID': current_test_id,
                'description': pred_desc
            })
            global_idx += 1

    # ==========================================
    # 4. SAVE
    # ==========================================
    df = pd.DataFrame(results)
    df.to_csv(SUBMISSION_CSV, index=False)
    print(f"\nSuccess! Reranked submission saved to {SUBMISSION_CSV}")
    print(df.head())

if __name__ == "__main__":
    generate_submission()

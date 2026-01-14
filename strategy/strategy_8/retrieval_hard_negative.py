import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure these utilities are available in your path
from utils.data_utils import (
    load_id2emb, load_descriptions_from_graphs, 
    PreprocessedGraphDataset, collate_fn
)

# Import the model class (adjust import path if your file structure changed)
from strategy.strategy_6.train_dual_tower import MolTransformerDual

# ==========================================
# CONFIGURATION (Matched to train_hard_negative.py)
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model Parameters
MODEL_PATH = "best_model_strategy_8.pt"
HIDDEN_DIM = 128
TEXT_DIM = 768
OUT_DIM = 768

# Data Paths
TRAIN_EMB_PATH = "embeddings/train_chembed_embeddings.csv"
VAL_EMB_PATH = "embeddings/validation_chembed_embeddings.csv"
TRAIN_GRAPHS_PATH = "data/train_graphs.pkl"
VAL_GRAPHS_PATH = "data/validation_graphs.pkl"
TEST_GRAPHS_PATH = "data/test_graphs.pkl"

# Output
SUBMISSION_CSV = "submission_strategy_8.csv"
BATCH_SIZE = 32

@torch.no_grad()
@torch.no_grad()
def generate_submission():
    print(f"Device: {DEVICE}")
    
    # 1. LOAD MODEL
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found. Run train_hard_negative.py first.")
        return

    model = MolTransformerDual(hidden=HIDDEN_DIM, text_dim=TEXT_DIM, out_dim=OUT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. PREPARE CANDIDATE LIBRARY (TRAIN + VAL)
    print("Loading Text Embeddings (ChemBed)...")
    train_emb = load_id2emb(TRAIN_EMB_PATH)
    val_emb = load_id2emb(VAL_EMB_PATH) if os.path.exists(VAL_EMB_PATH) else {}
    
    full_emb_dict = {**train_emb, **val_emb}
    candidate_ids = list(full_emb_dict.keys())
    print(f"Total Candidate Descriptions: {len(candidate_ids)}")

    print("Loading Original Text Descriptions...")
    train_desc = load_descriptions_from_graphs(TRAIN_GRAPHS_PATH)
    val_desc = load_descriptions_from_graphs(VAL_GRAPHS_PATH) if os.path.exists(VAL_GRAPHS_PATH) else {}
    full_id2desc = {**train_desc, **val_desc}

    # 3. LIBRARY UPGRADE
    print("Upgrading Text Library (Projecting to Shared Space)...")
    all_raw_embs = torch.stack([full_emb_dict[id_] for id_ in candidate_ids])
    
    candidate_pool_list = []
    chunk_size = 1024
    
    for i in range(0, len(all_raw_embs), chunk_size):
        batch_raw = all_raw_embs[i:i+chunk_size].to(DEVICE)
        batch_projected = model.forward_text(batch_raw)
        batch_norm = F.normalize(batch_projected, p=2, dim=-1)
        candidate_pool_list.append(batch_norm)
        
    candidate_pool = torch.cat(candidate_pool_list, dim=0) 
    print(f"Library Upgraded. Shape: {candidate_pool.size()}")

    # 4. PROCESS TEST MOLECULES
    print(f"Loading Test Graphs from {TEST_GRAPHS_PATH}...")
    
    # === FIX 1: Pass None explicitly to avoid KeyError ===
    test_ds = PreprocessedGraphDataset(TEST_GRAPHS_PATH, None) 
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Extract Test IDs
    if hasattr(test_ds, 'dataset'): 
        test_ids_list = [g.id for g in test_ds.dataset.graphs]
    else:
        test_ids_list = [g.id for g in test_ds.graphs]

    results = []
    global_idx = 0
    
    print("Running Inference on Test Set...")
    
    # === FIX 2: Handle single return value (graphs only) ===
    for batch in tqdm(test_dl):
        # If the collate function returns a tuple (graph, None) or just graph, handle both:
        if isinstance(batch, (list, tuple)):
            graphs = batch[0]
        else:
            graphs = batch
            
        graphs = graphs.to(DEVICE)
        bs = graphs.num_graphs
        
        # --- GRAPH ENCODING ---
        mol_vec = model.forward_graph(graphs)
        mol_vec = F.normalize(mol_vec, p=2, dim=-1)
        
        # --- SIMILARITY SEARCH ---
        sim_matrix = mol_vec @ candidate_pool.t()
        
        # Find Top 1
        _, top_indices = sim_matrix.topk(1, dim=1)
        top_indices = top_indices.cpu().numpy()
        
        # --- COLLECT RESULTS ---
        for i in range(bs):
            winner_idx = top_indices[i][0]
            pred_db_id = candidate_ids[winner_idx]
            pred_desc = full_id2desc[pred_db_id]
            current_test_id = test_ids_list[global_idx]
            
            results.append({
                'ID': current_test_id,
                'description': pred_desc
            })
            global_idx += 1

    # 5. SAVE SUBMISSION
    df = pd.DataFrame(results)
    df.to_csv(SUBMISSION_CSV, index=False)
    print(f"\nSuccess! Submission saved to {SUBMISSION_CSV}")
    print(df.head())
    
if __name__ == "__main__":
    generate_submission()

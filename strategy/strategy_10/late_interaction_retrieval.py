import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure these utilities are available in your path
from utils.data_utils import (
    load_id2emb, load_id2emb_colbert, load_descriptions_from_graphs, 
    PreprocessedGraphDataset, collate_fn
)

# Import the model class with ColBERT support
from strategy.strategy_10.mol_transformer_dual import MolTransformerDual, colbert_score

# ==========================================
# CONFIGURATION (Matched to train_hard_negative.py)
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model Parameters
MODEL_PATH = "/kaggle/working/ALTEGRAD-2025/model_colbert.pt"  
HIDDEN_DIM = 128
TEXT_DIM = 32 * 768  # ✅ Flattened ColBERT tokens (32 tokens × 768 dim)
OUT_DIM = 192  # ✅ Optimized dimension (MUST match training)
USE_COLBERT = True
NUM_TEXT_TOKENS = 32

# Data Paths - Using REAL ColBERT token embeddings
TRAIN_EMB_PATH = "/kaggle/working/ALTEGRAD-2025/train_chembed_colbert_embeddings.csv"
VAL_EMB_PATH = "/kaggle/working/ALTEGRAD-2025/validation_chembed_colbert_embeddings.csv"
TRAIN_GRAPHS_PATH = "/kaggle/input/molecular-data/train_graphs.pkl"  
VAL_GRAPHS_PATH = "/kaggle/input/molecular-data/validation_graphs.pkl"  
TEST_GRAPHS_PATH = "/kaggle/input/molecular-data/test_graphs.pkl"  

# Output
SUBMISSION_CSV = "submission_colbert.csv"
BATCH_SIZE = 32

@torch.no_grad()
def generate_submission():
    print(f"Device: {DEVICE}")
    
    # 1. LOAD MODEL
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found. Run train_late_interaction.py first.")
        return

    model = MolTransformerDual(
        hidden=HIDDEN_DIM, 
        text_dim=TEXT_DIM, 
        out_dim=OUT_DIM,
        use_colbert=USE_COLBERT,
        num_text_tokens=NUM_TEXT_TOKENS
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. PREPARE CANDIDATE LIBRARY (TRAIN + VAL)
    print("Loading Text Embeddings (ChemBed ColBERT Tokens)...")
    train_emb = load_id2emb_colbert(TRAIN_EMB_PATH, num_tokens=NUM_TEXT_TOKENS, hidden_dim=768)
    val_emb = load_id2emb_colbert(VAL_EMB_PATH, num_tokens=NUM_TEXT_TOKENS, hidden_dim=768) if os.path.exists(VAL_EMB_PATH) else {}
    
    full_emb_dict = {**train_emb, **val_emb}
    candidate_ids = list(full_emb_dict.keys())
    print(f"Total Candidate Descriptions: {len(candidate_ids)}")

    print("Loading Original Text Descriptions...")
    train_desc = load_descriptions_from_graphs(TRAIN_GRAPHS_PATH)
    val_desc = load_descriptions_from_graphs(VAL_GRAPHS_PATH) if os.path.exists(VAL_GRAPHS_PATH) else {}
    full_id2desc = {**train_desc, **val_desc}

    # 3. LIBRARY UPGRADE (ColBERT: Text Token Embeddings)
    print("Upgrading Text Library (Projecting to Shared Space with ColBERT)...")
    # Each embedding is [num_tokens, 768] for ColBERT
    # Flatten to [num_tokens * 768] for model input
    all_raw_embs = torch.stack([full_emb_dict[id_].flatten() for id_ in candidate_ids])
    
    candidate_pool_list = []
    chunk_size = 1024
    
    if USE_COLBERT:
        # Generate token embeddings for all candidate descriptions
        for i in range(0, len(all_raw_embs), chunk_size):
            batch_raw = all_raw_embs[i:i+chunk_size].to(DEVICE)
            batch_tokens = model.forward_text(batch_raw, return_tokens=True)
            # batch_tokens: [chunk_size, num_text_tokens, out_dim]
            candidate_pool_list.append(batch_tokens)
    else:
        for i in range(0, len(all_raw_embs), chunk_size):
            batch_raw = all_raw_embs[i:i+chunk_size].to(DEVICE)
            batch_projected = model.forward_text(batch_raw, return_tokens=False)
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
        if USE_COLBERT:
            mol_tokens, mol_mask = model.forward_graph(graphs, return_tokens=True)
            # mol_tokens: [bs, max_nodes, out_dim], mol_mask: [bs, max_nodes]
            
            # --- SIMILARITY SEARCH with ColBERT ---
            # Use colbert_score function for consistency with training
            # candidate_pool: [num_candidates, num_text_tokens, out_dim]
            # mol_tokens: [bs, max_nodes, out_dim]
            # Need to compute score for each molecule vs all candidates
            
            batch_size_mol = mol_tokens.size(0)
            num_candidates = candidate_pool.size(0)
            sim_matrix = torch.zeros(batch_size_mol, num_candidates, device=DEVICE)
            
            # Process each molecule individually to avoid OOM
            for mol_idx in range(batch_size_mol):
                # Get single molecule tokens: [1, max_nodes, out_dim]
                mol_single = mol_tokens[mol_idx:mol_idx+1]
                mask_single = mol_mask[mol_idx:mol_idx+1]
                
                # Expand to match candidates: [num_candidates, max_nodes, out_dim]
                mol_expanded = mol_single.expand(num_candidates, -1, -1)
                mask_expanded = mask_single.expand(num_candidates, -1)
                
                # Compute ColBERT scores: [num_candidates, num_candidates]
                # We only need the diagonal since mol is repeated for each candidate
                scores_full = colbert_score(candidate_pool, mol_expanded, mask_expanded)
                
                # Extract diagonal (each candidate vs this molecule)
                scores = torch.diag(scores_full)  # [num_candidates]
                sim_matrix[mol_idx] = scores
        else:
            mol_vec = model.forward_graph(graphs, return_tokens=False)
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

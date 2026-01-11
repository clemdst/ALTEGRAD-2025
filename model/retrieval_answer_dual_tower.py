import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import (
    load_id2emb, load_descriptions_from_graphs, 
    PreprocessedGraphDataset, collate_fn
)

# Import the model class and config from your training script
from train_dual_tower_neg_mining import (
    MolTransformerDual, DEVICE, 
    TRAIN_GRAPHS, VAL_GRAPHS, TEST_GRAPHS,
    TRAIN_EMB_CSV, VAL_EMB_CSV
)

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "dual_tower_best_temp.pt"
SUBMISSION_CSV = "submission_dual_tower_hard_best_temp_V1.csv"
BATCH_SIZE = 32

@torch.no_grad()
def generate_submission():
    print(f"Device: {DEVICE}")
    
    # 1. LOAD TEXT EMBEDDINGS FIRST (to detect dimension)
    print("Loading Text Embeddings (Train + Val)...")
    train_emb = load_id2emb(TRAIN_EMB_CSV)
    val_emb = load_id2emb(VAL_EMB_CSV) if os.path.exists(VAL_EMB_CSV) else {}
    
    # Merge Embeddings
    full_emb_dict = {**train_emb, **val_emb}
    candidate_ids = list(full_emb_dict.keys())
    print(f"Total Candidate Descriptions: {len(candidate_ids)}")
    
    # Detect embedding dimension from first sample
    sample_emb = next(iter(full_emb_dict.values()))
    text_embedding_dim = sample_emb.shape[0]
    print(f"Detected text embedding dimension: {text_embedding_dim}")
    
    # 2. LOAD MODEL
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print("Error: Model file not found. Run training first.")
        return

    # Initialize model structure with correct text_dim (must match training exactly)
    model = MolTransformerDual(
        hidden=128, 
        text_dim=text_embedding_dim,  # Use detected dimension
        out_dim=768,
        use_cross_attn=True  # Match training configuration
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Model loaded successfully (text_dim={text_embedding_dim} → out_dim=768)")

    # 3. Load Actual Descriptions (for the CSV output)
    print("Loading Text Descriptions...")
    train_desc = load_descriptions_from_graphs(TRAIN_GRAPHS)
    val_desc = load_descriptions_from_graphs(VAL_GRAPHS) if os.path.exists(VAL_GRAPHS) else {}
    full_id2desc = {**train_desc, **val_desc}

    # 4. LIBRARY UPGRADE (RE-EMBEDDING)
    # We must project the raw SciBERT embeddings through the trained Text Tower
    print("Upgrading Text Library (Projecting to Shared Space)...")
    
    # Convert dictionary to tensor for batch processing
    all_raw_embs = torch.stack([full_emb_dict[id_] for id_ in candidate_ids])
    
    candidate_pool_list = []
    # Process in chunks to avoid OOM
    chunk_size = 1024
    for i in range(0, len(all_raw_embs), chunk_size):
        batch_raw = all_raw_embs[i:i+chunk_size].to(DEVICE)
        
        # --- THE MAGIC STEP ---
        # Pass raw text through the Text Tower
        batch_projected = model.forward_text(batch_raw)
        
        # Normalize immediately for Cosine Similarity
        batch_norm = F.normalize(batch_projected, p=2, dim=-1)
        candidate_pool_list.append(batch_norm)
        
    candidate_pool = torch.cat(candidate_pool_list, dim=0) # [Total_Cands, 768]
    print(f"Library Upgraded. Shape: {candidate_pool.size()}")

    # 5. PROCESS TEST MOLECULES
    print("Loading Test Graphs...")
    test_ds = PreprocessedGraphDataset(TEST_GRAPHS)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Get Test IDs
    # Handling different dataset implementations (wrapper vs direct)
    if hasattr(test_ds, 'dataset'): 
        test_ids_list = [g.id for g in test_ds.dataset.graphs]
    else:
        test_ids_list = [g.id for g in test_ds.graphs]

    results = []
    global_idx = 0
    
    print("Running Inference...")
    for graphs in tqdm(test_dl):
        graphs = graphs.to(DEVICE)
        bs = graphs.num_graphs
        
        # --- GRAPH ENCODING ---
        # Pass graphs through the Graph Tower
        mol_vec = model.forward_graph(graphs)
        mol_vec = F.normalize(mol_vec, p=2, dim=-1)
        
        # --- SIMILARITY SEARCH ---
        # Dot product between [Batch, 768] and [Library, 768]
        sim_matrix = mol_vec @ candidate_pool.t()
        
        # Find best match (Top 1)
        _, top_indices = sim_matrix.topk(1, dim=1)
        top_indices = top_indices.cpu().numpy()
        
        # --- COLLECT RESULTS ---
        for i in range(bs):
            winner_idx = top_indices[i][0]
            
            # Map index back to real ID and Description
            pred_db_id = candidate_ids[winner_idx]
            pred_desc = full_id2desc[pred_db_id]
            current_test_id = test_ids_list[global_idx]
            
            results.append({
                'ID': current_test_id,
                'description': pred_desc
            })
            global_idx += 1

    # 6. SAVE SUBMISSION
    df = pd.DataFrame(results)
    df.to_csv(SUBMISSION_CSV, index=False)
    print(f"\nSuccess! Submission saved to {SUBMISSION_CSV}")
    print(df.head())

if __name__ == "__main__":
    generate_submission()

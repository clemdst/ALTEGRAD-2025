import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data_utils import (
    load_id2emb, load_descriptions_from_graphs, PreprocessedGraphDataset, collate_fn
)

from strategy.strategy_4.train_gine_rerank_matryoshka import (
    MolGINE, MolGINE_Residual_Reranker, DEVICE, 
    TRAIN_GRAPHS, VAL_GRAPHS, TEST_GRAPHS,  # Added VAL_GRAPHS
    TRAIN_EMB_CSV, VAL_EMB_CSV,             # Added VAL_EMB_CSV
    LimitedGraphDataset, TEST_MODE, N_SAMPLES
)

# ==========================================
# CONFIGURATION
# ==========================================
RETRIEVER_PATH = "models/model_strategy_3.pt"
RERANKER_PATH = "models/model_strategy_4.pt"

TOP_K = 50           
DIM_RETRIEVAL = 256  

def create_limited_emb_dict(full_emb_dict, n_samples):
    limited_dict = {}
    for i, (key, value) in enumerate(full_emb_dict.items()):
        if i >= n_samples: break
        limited_dict[key] = value
    return limited_dict

@torch.no_grad()
def generate_inference(retriever, reranker, test_data, combined_emb_dict, combined_id2desc, device, output_csv, test_mode=False, n_test_samples=None):
    print(f"Starting Inference [Retriever Dim: {DIM_RETRIEVAL} | Rerank Top: {TOP_K}]")
    
    # 1. Prepare COMPLETE Candidate Pool (Train + Val)
    candidate_ids = list(combined_emb_dict.keys())
    
    # Create tensor for search
    candidate_pool = torch.stack([combined_emb_dict[id_] for id_ in candidate_ids]).to(device)
    candidate_pool_retrieval = F.normalize(candidate_pool[:, :DIM_RETRIEVAL], dim=-1)
    
    print(f"Total Candidate Pool Size: {len(candidate_ids)} descriptions")

    # 2. Prepare Test Dataset
    full_test_ds = PreprocessedGraphDataset(test_data)
    if test_mode and n_test_samples is not None:
        test_ds = LimitedGraphDataset(full_test_ds, n_test_samples)
    else:
        test_ds = full_test_ds
        
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Track Test IDs
    if hasattr(test_ds, 'dataset'):
        source_graphs = [test_ds.dataset.graphs[i] for i in range(len(test_ds))]
    else:
        source_graphs = test_ds.graphs
    test_ids_list = [g.id for g in source_graphs]
    
    results = []
    global_idx = 0

    # 3. Inference Loop
    for graphs in tqdm(test_dl, desc="Processing Batches"):
        graphs = graphs.to(device)
        bs = graphs.num_graphs

        # --- STAGE 1: RETRIEVAL ---
        mol_raw = retriever(graphs)
        mol_vec_retrieval = F.normalize(mol_raw[:, :DIM_RETRIEVAL], dim=-1)
        
        sim_matrix = mol_vec_retrieval @ candidate_pool_retrieval.t()
        _, topk_indices = sim_matrix.topk(TOP_K, dim=1)

        # --- STAGE 2: RERANKING ---
        batch_candidates = candidate_pool[topk_indices] 
        rerank_scores = reranker(graphs, batch_candidates)
        
        best_local_indices = rerank_scores.argmax(dim=1).cpu().numpy()
        topk_indices = topk_indices.cpu().numpy()

        for i in range(bs):
            winner_k_idx = best_local_indices[i]
            winner_global_idx = topk_indices[i, winner_k_idx]
            
            # Look up from the COMBINED lists
            pred_db_id = candidate_ids[winner_global_idx]
            pred_desc = combined_id2desc[pred_db_id]
            current_test_id = test_ids_list[global_idx]
            
            results.append({'ID': current_test_id, 'description': pred_desc})
            global_idx += 1

    # 4. Save
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved submission to: {output_csv}")

def main():
    print(f"Device: {DEVICE}")
    
    # --- LOAD AND MERGE DATA ---
    print("Loading Train Embeddings...")
    train_emb = load_id2emb(TRAIN_EMB_CSV)
    print("Loading Validation Embeddings...")
    val_emb = load_id2emb(VAL_EMB_CSV)
    
    # MERGE DICTIONARIES
    if TEST_MODE:
        train_emb = create_limited_emb_dict(train_emb, N_SAMPLES)
        # In test mode we might skip merging val to keep it fast, or merge a small slice
        combined_emb = train_emb 
        n_test_samples = N_SAMPLES
        output_csv = "submission_test.csv"
    else:
        # Full Merge
        combined_emb = {**train_emb, **val_emb}
        n_test_samples = None
        output_csv = "results/strategy_4_test_descriptions.csv"

    print("Loading Descriptions (Train + Val)...")
    train_desc = load_descriptions_from_graphs(TRAIN_GRAPHS)
    val_desc = load_descriptions_from_graphs(VAL_GRAPHS)
    combined_desc = {**train_desc, **val_desc}
    
    # --- LOAD MODELS ---
    emb_dim = len(next(iter(train_emb.values())))
    
    retriever = MolGINE(out_dim=emb_dim).to(DEVICE)
    retriever.load_state_dict(torch.load(RETRIEVER_PATH, map_location=DEVICE))
    retriever.eval()
    
    reranker = MolGINE_Residual_Reranker(hidden=128, text_dim=emb_dim).to(DEVICE)
    reranker.load_state_dict(torch.load(RERANKER_PATH, map_location=DEVICE))
    reranker.eval()
    
    # --- RUN ---
    generate_inference(
        retriever=retriever,
        reranker=reranker,
        test_data=TEST_GRAPHS,
        combined_emb_dict=combined_emb,  # PASS MERGED DICT
        combined_id2desc=combined_desc,  # PASS MERGED DESC
        device=DEVICE,
        output_csv=output_csv,
        test_mode=TEST_MODE,
        n_test_samples=n_test_samples
    )

if __name__ == "__main__":
    main()

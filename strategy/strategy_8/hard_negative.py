import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs
from tqdm import tqdm
import os

# --- CONFIGURATION ---
# Check these paths match your Kaggle environment exactly
TRAIN_GRAPHS_PATH = "data/train_graphs.pkl" 
OUTPUT_FILE = "strategy/strategy_8/hard_negatives.npy"

# --- 1. Robust Reconstruction ---
def reconstruct_mol_safe(data):
    """
    Attempts to reconstruct a molecule. Returns None if it fails.
    """
    try:
        mol = Chem.RWMol()
        node_features = data.x.cpu().numpy()
        for i in range(len(node_features)):
            atomic_num = int(node_features[i, 0])
            mol.AddAtom(Chem.Atom(atomic_num))
        
        edge_index = data.edge_index.cpu().numpy()
        for k in range(edge_index.shape[1]):
            i, j = edge_index[0, k], edge_index[1, k]
            if i < j:
                mol.AddBond(int(i), int(j), Chem.rdchem.BondType.SINGLE)
        
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
        return mol
    except:
        return None

# --- 2. Main Generation Script ---
def generate_robust_negatives():
    print(f"Loading graphs from {TRAIN_GRAPHS_PATH}...")
    
    if not os.path.exists(TRAIN_GRAPHS_PATH):
        raise FileNotFoundError(f"Cannot find file at {TRAIN_GRAPHS_PATH}. Check your paths!")

    # FIX: Use pandas to read the pickle (handles protocols better than standard pickle)
    try:
        graphs = pd.read_pickle(TRAIN_GRAPHS_PATH)
    except Exception as e:
        print(f"Pandas load failed: {e}. Trying standard pickle...")
        import pickle
        with open(TRAIN_GRAPHS_PATH, 'rb') as f:
            graphs = pickle.load(f)
            
    print(f"Successfully loaded {len(graphs)} graphs.")
    
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    
    print("Generating fingerprints (Robust Mode)...")
    # This list maps 'search_index' -> 'original_dataset_index'
    valid_indices = []
    valid_fps = []

    for idx, g in enumerate(tqdm(graphs)):
        mol = reconstruct_mol_safe(g)
        if mol:
            fp = mfpgen.GetFingerprint(mol)
            valid_fps.append(fp)
            valid_indices.append(idx)
    
    # Pre-allocate the matrix with default random indices (safe fallback)
    n_total = len(graphs)
    hard_indices_matrix = np.random.randint(0, n_total, size=(n_total, 50))
    
    print(f"Computing Similarity for {len(valid_fps)} valid molecules...")
    
    # We only compute similarity for the valid molecules
    for i, fp_query in enumerate(tqdm(valid_fps)):
        # 1. Compute similarity against all other valid fingerprints
        sims = DataStructs.BulkTanimotoSimilarity(fp_query, valid_fps)
        
        # 2. Get top 51 indices (including self)
        # Note: These are indices in the 'valid_fps' list, NOT the original list
        top_k_valid_indices = np.argsort(sims)[-51:]
        
        # 3. Map back to original dataset indices
        original_idx_of_query = valid_indices[i]
        
        found_neighbors = []
        for valid_idx in reversed(top_k_valid_indices):
            real_idx = valid_indices[valid_idx]
            if real_idx != original_idx_of_query:
                found_neighbors.append(real_idx)
            if len(found_neighbors) == 50:
                break
        
        # 4. Store in the matrix at the correct row
        if len(found_neighbors) > 0:
            # If we found fewer than 50 (rare), fill remainder with randoms
            while len(found_neighbors) < 50:
                found_neighbors.append(np.random.choice(valid_indices))
            
            hard_indices_matrix[original_idx_of_query] = np.array(found_neighbors)

    np.save(OUTPUT_FILE, hard_indices_matrix)
    print(f"Done! Saved {OUTPUT_FILE} with shape {hard_indices_matrix.shape}")

if __name__ == "__main__":
    generate_robust_negatives()

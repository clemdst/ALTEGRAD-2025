import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.data_utils import (
    load_id2emb, load_descriptions_from_graphs, PreprocessedGraphDataset, collate_fn
)

from train_gineconv import (
    MolGNN, DEVICE, TRAIN_GRAPHS, TEST_GRAPHS, TRAIN_EMB_CSV, 
    LimitedGraphDataset, TEST_MODE, N_SAMPLES
)


def create_limited_emb_dict(full_emb_dict, n_samples):
    """Create a limited embedding dictionary with only the first n_samples"""
    limited_dict = {}
    for i, (key, value) in enumerate(full_emb_dict.items()):
        if i >= n_samples:
            break
        limited_dict[key] = value
    return limited_dict


@torch.no_grad()
def retrieve_descriptions(model, train_data, test_data, train_emb_dict, device, output_csv, test_mode=False, n_test_samples=None):
    """
    Args:
        model: Trained GNN model
        train_data: Path to train preprocessed graphs
        test_data: Path to test preprocessed graphs
        train_emb_dict: Dictionary mapping train IDs to text embeddings
        device: Device to run on
        output_csv: Path to save retrieved descriptions
        test_mode: Whether to limit the number of test samples
        n_test_samples: Number of test samples to process (if test_mode=True)
    """
    train_id2desc = load_descriptions_from_graphs(train_data)
    
    train_ids = list(train_emb_dict.keys())
    train_embs = torch.stack([train_emb_dict[id_] for id_ in train_ids]).to(device)
    train_embs = F.normalize(train_embs, dim=-1)
    
    print(f"Train set size: {len(train_ids)}")
    
    # Create test dataset (with potential limitation)
    full_test_ds = PreprocessedGraphDataset(test_data)
    
    if test_mode and n_test_samples is not None:
        test_ds = LimitedGraphDataset(full_test_ds, n_test_samples)
        print(f"Limited test dataset to {len(test_ds)} samples")
    else:
        test_ds = full_test_ds
        print(f"Using full test dataset with {len(test_ds)} samples")
    
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    test_mol_embs = []
    test_ids_ordered = []
    for graphs in test_dl:
        graphs = graphs.to(device)
        mol_emb = model(graphs)
        test_mol_embs.append(mol_emb)
        batch_size = graphs.num_graphs
        start_idx = len(test_ids_ordered)
        
        # Get IDs from the limited dataset
        if test_mode and n_test_samples is not None:
            # For limited dataset, get IDs from the underlying full dataset
            end_idx = min(start_idx + batch_size, len(test_ds))
            for j in range(start_idx, end_idx):
                test_ids_ordered.append(test_ds.dataset.ids[j])
        else:
            test_ids_ordered.extend(test_ds.ids[start_idx:start_idx + batch_size])
    
    test_mol_embs = torch.cat(test_mol_embs, dim=0)
    print(f"Encoded {test_mol_embs.size(0)} test molecules")
    
    similarities = test_mol_embs @ train_embs.t()
    
    most_similar_indices = similarities.argmax(dim=-1).cpu()
    
    results = []
    for i, test_id in enumerate(test_ids_ordered):
        train_idx = most_similar_indices[i].item()
        retrieved_train_id = train_ids[train_idx]
        retrieved_desc = train_id2desc[retrieved_train_id]
        
        results.append({
            'ID': test_id,
            'description': retrieved_desc
        })
        
        if i < 5:
            print(f"\nTest ID {test_id}: Retrieved from train ID {retrieved_train_id}")
            print(f"Description: {retrieved_desc[:150]}...")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"\n{'='*80}")
    print(f"Saved {len(results)} retrieved descriptions to: {output_csv}")
    
    return results_df


def main():
    print(f"Device: {DEVICE}")
    
    if TEST_MODE:
        print(f"Running in TEST MODE with {N_SAMPLES} samples")
        output_csv = f"test_retrieved_descriptions_limited_{N_SAMPLES}.csv"
        # Use limited embeddings for train set too
        full_train_emb = load_id2emb(TRAIN_EMB_CSV)
        train_emb = create_limited_emb_dict(full_train_emb, N_SAMPLES)
        n_test_samples = N_SAMPLES // 2  # Use fewer test samples
    else:
        output_csv = "results/strategy_2_test_descriptions.csv"
        train_emb = load_id2emb(TRAIN_EMB_CSV)
        n_test_samples = None

    model_path = "models/model_strategy_2.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint '{model_path}' not found.")
        print("Please train a model first using train_gcn.py")
        return
    
    if not os.path.exists(TEST_GRAPHS):
        print(f"Error: Preprocessed graphs not found at {TEST_GRAPHS}")
        return
    
    emb_dim = len(next(iter(train_emb.values())))
    
    model = MolGNN(out_dim=emb_dim).to(DEVICE)
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    retrieve_descriptions(
        model=model,
        train_data=TRAIN_GRAPHS,
        test_data=TEST_GRAPHS,
        train_emb_dict=train_emb,
        device=DEVICE,
        output_csv=output_csv,
        test_mode=TEST_MODE,
        n_test_samples=n_test_samples
    )


if __name__ == "__main__":
    main()

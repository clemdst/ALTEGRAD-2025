#!/usr/bin/env python3
"""Generate SciBERT embeddings with Mean Pooling."""

import pickle
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from torch.utils.data import DataLoader

# ==============================
# IMPROVEMENT 1: Better Config
# ==============================
MAX_TOKEN_LENGTH = 256  # Increased from 128
BATCH_SIZE = 32         # Process in batches for speed

MODEL_NAME = 'allenai/scibert_scivocab_uncased' 

def mean_pooling(model_output, attention_mask):
    """
    Mean Pooling - Take attention mask into account for correct averaging
    """
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    
    # Sum embeddings of valid tokens
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    
    # Count valid tokens (avoid division by zero)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    return sum_embeddings / sum_mask

def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print(f"Model loaded on: {device}")

    for split in ['train', 'validation']:
        print(f"\nProcessing {split}...")
        
        # Load graphs
        pkl_path = f'/kaggle/input/data-challenge-altegrad/{split}_graphs.pkl'
        with open(pkl_path, 'rb') as f:
            graphs = pickle.load(f)
        
        # Extract ID and Description
        data_items = []
        for g in graphs:
            data_items.append({'id': g.id, 'text': g.description})
            
        print(f"Loaded {len(data_items)} items. Starting batch processing...")

        all_ids = []
        all_embeddings = []

        # Process in batches
        for i in tqdm(range(0, len(data_items), BATCH_SIZE)):
            batch = data_items[i : i + BATCH_SIZE]
            batch_texts = [item['text'] for item in batch]
            batch_ids = [item['id'] for item in batch]
            
            # Tokenize
            encoded_input = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=MAX_TOKEN_LENGTH, 
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {k: v.to(device) for k, v in encoded_input.items()}

            # Generate Embeddings
            with torch.no_grad():
                model_output = model(**inputs)

            
            embeddings = mean_pooling(model_output, inputs['attention_mask'])
            
            embeddings = embeddings.cpu().numpy()
            
            all_ids.extend(batch_ids)
            all_embeddings.extend(embeddings)

        # Save to CSV
        print("Formatting and saving...")
        # Convert numpy arrays to comma-separated strings
        str_embeddings = [','.join(map(str, emb)) for emb in all_embeddings]
        
        result = pd.DataFrame({
            'ID': all_ids,
            'embedding': str_embeddings
        })

        output_path = f'./embeddings/{split}_scibert_embeddings.csv'
        result.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()

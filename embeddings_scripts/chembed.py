#!/usr/bin/env python3
"""Generate ChEmbed embeddings with Mean Pooling."""

import pickle
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

# ==============================
# CONFIGURATION
# ==============================
# UPDATED: Switching to ChEmbed (State-of-the-Art for Chemical Text)
MODEL_NAME = 'BASF-AI/ChEmbed-base' 

# Config
MAX_TOKEN_LENGTH = 512  # Increased to 512 to capture full descriptions
BATCH_SIZE = 32         # Adjust to 16 if you get GPU OOM errors

# Output Directory
OUTPUT_DIR = './embeddings/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    print(f"Loading Model: {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"Error loading {MODEL_NAME}. Please check internet connection or model name.")
        raise e
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print(f"Model loaded on: {device}")

    for split in ['train', 'validation']:
        print(f"\nProcessing {split} graphs...")
        
        # Load graphs
        pkl_path = f'.data/{split}_graphs.pkl'
        if not os.path.exists(pkl_path):
            print(f"Warning: File not found {pkl_path}, skipping...")
            continue
            
        with open(pkl_path, 'rb') as f:
            graphs = pickle.load(f)
        
        # Extract ID and Description
        data_items = []
        for g in graphs:
            # Ensure description is a string
            desc = str(g.description) if hasattr(g, 'description') and g.description else ""
            data_items.append({'id': g.id, 'text': desc})
            
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

            # Mean Pooling
            embeddings = mean_pooling(model_output, inputs['attention_mask'])
            
            # Move to CPU for saving
            embeddings = embeddings.cpu().numpy()
            
            all_ids.extend(batch_ids)
            all_embeddings.extend(embeddings)

        # Save to CSV
        print(f"Formatting {len(all_embeddings)} embeddings...")
        
        # Convert numpy arrays to string format expected by your data_utils
        # Format: "[0.123, 0.456, ...]" (matches previous CSV style if needed) or simple comma sep
        # Using simple comma separated values inside the cell as per previous script
        str_embeddings = [','.join(map(str, emb)) for emb in all_embeddings]
        
        result = pd.DataFrame({
            'ID': all_ids,
            'embedding': str_embeddings
        })
        
        # CHANGED: Output filename to reflect new model
        output_filename = f'{split}_chembed_embeddings.csv'
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        result.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()

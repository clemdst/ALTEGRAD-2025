#!/usr/bin/env python3
"""Generate ChEmbed embeddings with ColBERT-style token-level representations.

ColBERT Mode (USE_COLBERT=True):
    - Generates token-level embeddings: [num_tokens, hidden_dim]
    - Each description is represented by NUM_TOKENS_COLBERT token vectors
    - Stored flattened as [num_tokens * hidden_dim] in CSV
    - To use: Reshape loaded embedding to [NUM_TOKENS_COLBERT, hidden_dim]
    
Traditional Mode (USE_COLBERT=False):
    - Generates mean-pooled embeddings: [hidden_dim]
    - Each description is represented by a single vector
    - Standard for dual-tower retrieval
    
Usage:
    1. Set USE_COLBERT=True for ColBERT mode
    2. Run: python embeddings_scripts/chembed_copy.py
    3. Load with load_id2emb_colbert() for token embeddings
"""

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
MODEL_NAME = 'BASF-AI/ChEmbed-full' 

# Config
MAX_TOKEN_LENGTH = 128  # ColBERT typically uses shorter sequences for efficiency
NUM_TOKENS_COLBERT = 32  # Number of tokens to keep for ColBERT (fixed-length)
BATCH_SIZE = 16         # Smaller batch size due to token-level storage
USE_COLBERT = True      # Set to False for traditional mean pooling

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

def colbert_token_embeddings(model_output, attention_mask, num_tokens=32):
    """
    ColBERT-style token embeddings - Keep top-k token representations
    
    Args:
        model_output: Transformer model output
        attention_mask: Attention mask [batch_size, seq_len]
        num_tokens: Number of tokens to keep (fixed length)
    
    Returns:
        Token embeddings [batch_size, num_tokens, hidden_dim]
    """
    token_embeddings = model_output.last_hidden_state  # [batch_size, seq_len, hidden_dim]
    batch_size, seq_len, hidden_dim = token_embeddings.shape
    
    # Normalize token embeddings
    token_embeddings = torch.nn.functional.normalize(token_embeddings, p=2, dim=-1)
    
    # Strategy 1: Take first num_tokens tokens (excluding padding)
    # This is simple and works well for ColBERT
    result = torch.zeros(batch_size, num_tokens, hidden_dim, device=token_embeddings.device)
    
    for i in range(batch_size):
        # Get valid tokens (non-padded)
        valid_mask = attention_mask[i].bool()
        valid_tokens = token_embeddings[i][valid_mask]  # [num_valid, hidden_dim]
        
        # Take first num_tokens or pad if fewer
        num_valid = valid_tokens.shape[0]
        if num_valid >= num_tokens:
            result[i] = valid_tokens[:num_tokens]
        else:
            # Pad with zeros if fewer tokens
            result[i, :num_valid] = valid_tokens
            # Remaining tokens stay as zeros
    
    return result

def main():
    print(f"Loading Model: {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME,trust_remote_code=True)
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
        pkl_path = f'/kaggle/input/molecular-data/{split}_graphs.pkl'
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

            # Choose embedding strategy
            if USE_COLBERT:
                # ColBERT: Token-level embeddings [batch_size, num_tokens, hidden_dim]
                embeddings = colbert_token_embeddings(
                    model_output, 
                    inputs['attention_mask'],
                    num_tokens=NUM_TOKENS_COLBERT
                )
                # Move to CPU and reshape to [batch_size, num_tokens * hidden_dim]
                embeddings = embeddings.cpu().numpy()
                # Flatten for storage: [batch_size, num_tokens * hidden_dim]
                embeddings = embeddings.reshape(embeddings.shape[0], -1)
            else:
                # Traditional: Mean Pooling [batch_size, hidden_dim]
                embeddings = mean_pooling(model_output, inputs['attention_mask'])
                embeddings = embeddings.cpu().numpy()
            
            all_ids.extend(batch_ids)
            all_embeddings.extend(embeddings)

        # Save to CSV
        print(f"Formatting {len(all_embeddings)} embeddings...")
        
        # Convert numpy arrays to string format
        # For ColBERT: This will be flattened [num_tokens * hidden_dim] vector
        # For Traditional: This will be [hidden_dim] vector
        str_embeddings = [','.join(map(str, emb)) for emb in all_embeddings]
        
        result = pd.DataFrame({
            'ID': all_ids,
            'embedding': str_embeddings
        })
        
        # Output filename reflects mode
        mode_suffix = 'colbert' if USE_COLBERT else 'meanpool'
        output_filename = f'{split}_chembed_{mode_suffix}_embeddings.csv'
        result.to_csv(output_filename, index=False)
        print(f"Saved to {output_filename}")
        
        if USE_COLBERT:
            print(f"  → ColBERT mode: {NUM_TOKENS_COLBERT} tokens per description")
            print(f"  → Embedding shape per sample: [{NUM_TOKENS_COLBERT} × hidden_dim]")
        else:
            print(f"  → Mean pooling mode: Single vector per description")

if __name__ == "__main__":
    main()

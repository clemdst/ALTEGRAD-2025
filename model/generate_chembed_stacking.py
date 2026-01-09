#!/usr/bin/env python3
"""
Generate STACKED embeddings with multiple models and pooling strategies.
Uses: ChEmbed + PubMedBERT with Mean Pooling + Attention Pooling
"""

import pickle
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

# ==============================
# CONFIGURATION
# ==============================
# Stacking Configuration: 2 Models × 2 Pooling Methods = 4 embeddings
MODELS_CONFIG = [
    {
        'name': 'BASF-AI/ChEmbed-full' ,
        'label': 'chembed'
    },
    {
        'name': 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
        'label': 'pubmedbert'
    }
]

POOLING_METHODS = ['mean', 'attention']  # Two pooling strategies

# Config
MAX_TOKEN_LENGTH = 512
BATCH_SIZE = 16  # Reduced for memory efficiency with multiple models
USE_FP16 = True  # Mixed precision for speed

# Output Directory
OUTPUT_DIR = './'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# POOLING STRATEGIES
# ==============================

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


def attention_pooling(model_output, attention_mask):
    """
    Attention-Weighted Pooling
    Uses attention weights from the last layer to pool token embeddings
    Falls back to mean pooling if attention not available
    """
    token_embeddings = model_output.last_hidden_state  # [B, Seq, D]
    
    # Check if attention outputs are available
    if hasattr(model_output, 'attentions') and model_output.attentions is not None and len(model_output.attentions) > 0:
        # Get attention from last layer
        last_attn = model_output.attentions[-1]  # [B, Heads, Seq, Seq]
        
        # Average across all attention heads
        attn_weights = last_attn.mean(dim=1)  # [B, Seq, Seq]
        
        # Use attention to [CLS] token (first token) as importance scores
        cls_attn = attn_weights[:, 0, :]  # [B, Seq]
        
        # Apply attention mask to ignore padding tokens
        cls_attn = cls_attn * attention_mask.float()
        
        # Normalize attention weights
        cls_attn_sum = cls_attn.sum(dim=1, keepdim=True)
        cls_attn = cls_attn / (cls_attn_sum + 1e-9)
        
        # Apply attention weights to token embeddings
        cls_attn = cls_attn.unsqueeze(-1)  # [B, Seq, 1]
        weighted_embeddings = (token_embeddings * cls_attn).sum(dim=1)  # [B, D]
        
        return weighted_embeddings
    else:
        # Fallback to mean pooling if attention not available
        return mean_pooling(model_output, attention_mask)


POOLING_FUNCTIONS = {
    'mean': mean_pooling,
    'attention': attention_pooling
}

# ==============================
# MODEL LOADING
# ==============================

def load_models():
    """Load all models with their tokenizers"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Mixed Precision (FP16): {USE_FP16 and torch.cuda.is_available()}")
    print("="*80)
    
    models_data = []
    
    for config in MODELS_CONFIG:
        model_name = config['name']
        label = config['label']
        
        print(f"\nLoading {label}: {model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                model_name,
                output_hidden_states=True,  # For potential layer-wise pooling
                output_attentions=True,      # Required for attention pooling
                trust_remote_code=True       # Allow custom code execution
            )
            model = model.to(device)
            model.eval()
            
            # Enable FP16 if requested and available
            if USE_FP16 and torch.cuda.is_available():
                model = model.half()
            
            models_data.append({
                'model': model,
                'tokenizer': tokenizer,
                'label': label,
                'device': device
            })
            print(f"  ✓ {label} loaded successfully")
            
        except Exception as e:
            print(f"  ✗ Failed to load {label}: {e}")
            raise e
    
    print("\n" + "="*80)
    return models_data


# ==============================
# EMBEDDING GENERATION
# ==============================

@torch.no_grad()
def generate_stacked_embeddings(batch_texts, models_data):
    """
    Generate stacked embeddings from multiple models and pooling methods
    Returns: Concatenated embeddings [model1_mean, model1_attention, model2_mean, model2_attention]
    """
    all_embeddings = []
    
    for model_data in models_data:
        model = model_data['model']
        tokenizer = model_data['tokenizer']
        device = model_data['device']
        label = model_data['label']
        
        # Tokenize
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in encoded.items()}
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=USE_FP16 and torch.cuda.is_available()):
            outputs = model(**inputs)
        
        # Apply each pooling method
        for pooling_name in POOLING_METHODS:
            pooling_fn = POOLING_FUNCTIONS[pooling_name]
            embeddings = pooling_fn(outputs, inputs['attention_mask'])
            
            # Normalize embeddings (L2 normalization for cosine similarity)
            embeddings = F.normalize(embeddings.float(), p=2, dim=-1)
            
            # Move to CPU and convert to numpy
            embeddings_np = embeddings.cpu().numpy()
            all_embeddings.append(embeddings_np)
    
    # Stack all embeddings horizontally (concatenate features)
    # Shape: [batch_size, total_embedding_dim]
    stacked = np.concatenate(all_embeddings, axis=1)
    
    return stacked


# ==============================
# MAIN PROCESSING
# ==============================

def main():
    print("="*80)
    print("STACKED EMBEDDINGS GENERATION")
    print(f"Models: {len(MODELS_CONFIG)} ({', '.join([m['label'] for m in MODELS_CONFIG])})")
    print(f"Pooling Methods: {len(POOLING_METHODS)} ({', '.join(POOLING_METHODS)})")
    print(f"Total Embeddings per Sample: {len(MODELS_CONFIG) * len(POOLING_METHODS)}")
    print("="*80)
    
    # Load all models
    models_data = load_models()

    for split in ['train', 'validation']:
        print(f"\n{'='*80}")
        print(f"Processing {split.upper()} split")
        print('='*80)
        
        # Load graphs
        pkl_path = f'/kaggle/input/molecular-data/{split}_graphs.pkl' 
        if not os.path.exists(pkl_path):
            print(f"⚠ Warning: File not found {pkl_path}, skipping...")
            continue
            
        with open(pkl_path, 'rb') as f:
            graphs = pickle.load(f)
        
        print(f"Loaded {len(graphs)} graphs")
        
        # Extract ID and Description
        data_items = []
        for g in graphs:
            # Ensure description is a string
            desc = str(g.description) if hasattr(g, 'description') and g.description else ""
            data_items.append({'id': g.id, 'text': desc})
        
        all_ids = []
        all_embeddings = []
        
        # Process in batches
        print(f"\nGenerating stacked embeddings (batch size={BATCH_SIZE})...")
        for i in tqdm(range(0, len(data_items), BATCH_SIZE), desc=f"{split}"):
            batch = data_items[i : i + BATCH_SIZE]
            batch_texts = [item['text'] for item in batch]
            batch_ids = [item['id'] for item in batch]
            
            try:
                # Generate stacked embeddings
                stacked_embeddings = generate_stacked_embeddings(batch_texts, models_data)
                
                all_ids.extend(batch_ids)
                all_embeddings.extend(stacked_embeddings)
                
            except Exception as e:
                print(f"\n⚠ Error processing batch {i//BATCH_SIZE + 1}: {e}")
                continue
        
        # Save to CSV
        print(f"\nFormatting and saving {len(all_embeddings)} stacked embeddings...")
        
        # Convert numpy arrays to string format (comma-separated values)
        str_embeddings = [','.join(map(str, emb)) for emb in all_embeddings]
        
        result = pd.DataFrame({
            'ID': all_ids,
            'embedding': str_embeddings
        })
        
        # Output filename reflects stacking approach
        output_filename = f'{split}_stacked_embeddings.csv'
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        result.to_csv(output_path, index=False)
        print(f"✓ Saved to: {output_path}")
        
        # Print embedding statistics
        emb_matrix = np.array(all_embeddings)
        print(f"\nEmbedding Statistics:")
        print(f"  Shape: {emb_matrix.shape}")
        print(f"  Dimension: {emb_matrix.shape[1]} (per model: ~{emb_matrix.shape[1] // (len(MODELS_CONFIG) * len(POOLING_METHODS))})")
        print(f"  Mean norm: {np.linalg.norm(emb_matrix, axis=1).mean():.4f}")
        print(f"  Std norm: {np.linalg.norm(emb_matrix, axis=1).std():.4f}")
        
        # Breakdown by model and pooling
        emb_dim_per_component = emb_matrix.shape[1] // (len(MODELS_CONFIG) * len(POOLING_METHODS))
        idx = 0
        print(f"\n  Component Breakdown:")
        for model_cfg in MODELS_CONFIG:
            for pooling in POOLING_METHODS:
                start_idx = idx * emb_dim_per_component
                end_idx = (idx + 1) * emb_dim_per_component
                component = emb_matrix[:, start_idx:end_idx]
                print(f"    {model_cfg['label']}_{pooling}: dims [{start_idx}:{end_idx}], mean_norm={np.linalg.norm(component, axis=1).mean():.4f}")
                idx += 1
    
    print("\n" + "="*80)
    print("✓ STACKED EMBEDDINGS GENERATION COMPLETE!")
    print(f"✓ Output files: train_stacked_embeddings.csv, validation_stacked_embeddings.csv")
    print("="*80)

if __name__ == "__main__":
    main()

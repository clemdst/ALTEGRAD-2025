import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import pandas as pd
import numpy as np

# Metrics
import sacrebleu
from bert_score import score as bert_score_func

# PyG Imports
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import TransformerConv

# Transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Import Utils & Previous Models
from utils.data_utils import PreprocessedGraphDataset, collate_fn, x_map, e_map
from strategy.strategy_6.train_dual_tower import MolTransformerDual 

# =========================================================
# CONFIGURATION
# =========================================================
TRAIN_GRAPHS  = "data/train_graphs.pkl"
VAL_GRAPHS    = "data/validation_graphs.pkl"
RETRIEVER_PATH = "models/model_strategy_8.pt" 

OUTPUT_DIR    = "models/"
MODEL_PATH    = os.path.join(OUTPUT_DIR, "models/model_strategy_9.pt")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MOLT5_MODEL   = "laituan245/molt5-small"
BATCH_SIZE    = 16
EPOCHS        = 15
LR            = 1e-4
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K         = 3 

# =========================================================
# 1. THE GENERATOR (Graph + Hints -> Text)
# =========================================================
class MolGNN_RAG(nn.Module):
    def __init__(self, hidden=128, layers=3, heads=4, text_model="laituan245/molt5-small"):
        super().__init__()
        
        # --- GRAPH ENCODER ---
        self.node_emb = nn.ModuleList([nn.Embedding(len(x_map[key]), hidden) for key in x_map])
        self.node_proj = nn.Linear(hidden * len(x_map), hidden)
        
        self.edge_emb = nn.ModuleList([nn.Embedding(len(e_map[key]), hidden) for key in e_map])
        self.edge_proj = nn.Linear(hidden * len(e_map), hidden)
        
        self.convs = nn.ModuleList()
        for _ in range(layers):
            self.convs.append(TransformerConv(
                in_channels=hidden, out_channels=hidden//heads, heads=heads,
                dropout=0.1, edge_dim=hidden, beta=True
            ))
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        
        # --- RAG BRIDGE ---
        self.t5 = T5ForConditionalGeneration.from_pretrained(text_model)
        self.graph_proj = nn.Linear(hidden, self.t5.config.d_model)

    def encode_graph(self, batch):
        # Embed Nodes
        node_feats = [emb(batch.x[:, i]) for i, emb in enumerate(self.node_emb)]
        x = self.node_proj(torch.cat(node_feats, dim=-1))
        # Embed Edges
        edge_feats = [emb(batch.edge_attr[:, i]) for i, emb in enumerate(self.edge_emb)]
        edge_attr = self.edge_proj(torch.cat(edge_feats, dim=-1))
        # GNN
        for conv, ln in zip(self.convs, self.layer_norms):
            x = F.relu(ln(x + conv(x, batch.edge_index, edge_attr)))
            
        # To Dense Sequence [Batch, Max_Nodes, Dim]
        x_dense, mask = to_dense_batch(x, batch.batch)
        return self.graph_proj(x_dense), mask

    def forward(self, batch_graph, hint_input_ids, hint_mask, labels=None):
        graph_embeds, graph_mask = self.encode_graph(batch_graph)
        hint_embeds = self.t5.shared(hint_input_ids)
        
        # Concatenate: [HINTS | GRAPH]
        inputs_embeds = torch.cat([hint_embeds, graph_embeds], dim=1)
        attention_mask = torch.cat([hint_mask, graph_mask], dim=1)
        
        return self.t5(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels).loss

    def generate(self, batch_graph, hint_input_ids, hint_mask, max_new_tokens=100):
        """
        Inference Step
        """
        graph_embeds, graph_mask = self.encode_graph(batch_graph)
        hint_embeds = self.t5.shared(hint_input_ids)
        
        inputs_embeds = torch.cat([hint_embeds, graph_embeds], dim=1)
        attention_mask = torch.cat([hint_mask, graph_mask], dim=1)
        
        return self.t5.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_length=max_new_tokens,
            num_beams=5,
            early_stopping=True
        )

# =========================================================
# 2. RETRIEVAL MANAGER
# =========================================================
class RetrievalManager:
    def __init__(self, model_path, train_loader, device):
        self.device = device
        print("Loading Retriever (Dual Tower)...")
        self.retriever = MolTransformerDual(hidden=128, text_dim=768, out_dim=768).to(device)
        self.retriever.load_state_dict(torch.load(model_path, map_location=device))
        self.retriever.eval()
        
        print("Building Retrieval Index (this takes 1 min)...")
        self.index_vecs = []
        self.index_texts = []
        
        for batch in tqdm(train_loader, desc="Indexing"):
            if isinstance(batch, tuple):
                graphs, text_embs = batch
                with torch.no_grad():
                    vecs = self.retriever.forward_text(text_embs.to(device))
                    self.index_vecs.append(F.normalize(vecs, dim=-1).cpu())
                for g in graphs.to_data_list():
                    self.index_texts.append(g.description)
        
        self.index_vecs = torch.cat(self.index_vecs, dim=0).to(device)
        print(f"Index Built: {self.index_vecs.shape}")

    def get_hints(self, batch_graph, k=3, exclude_self=True):
        with torch.no_grad():
            query_vecs = self.retriever.forward_graph(batch_graph.to(self.device))
            query_vecs = F.normalize(query_vecs, dim=-1)
            sim_matrix = query_vecs @ self.index_vecs.T
            
            if exclude_self:
                vals, inds = sim_matrix.topk(k+1, dim=1)
            else:
                vals, inds = sim_matrix.topk(k, dim=1)
                
            hints_batch = []
            for r in range(len(batch_graph)):
                candidates = inds[r].cpu().tolist()
                valid_hints = []
                for idx in candidates:
                    candidate_text = self.index_texts[idx]
                    target_text = batch_graph[r].description
                    if exclude_self and (candidate_text == target_text):
                        continue
                    valid_hints.append(candidate_text)
                    if len(valid_hints) == k: break
                
                prompt = " ".join([f"Hint: {h}" for h in valid_hints])
                hints_batch.append(prompt)
            return hints_batch

# =========================================================
# 3. EVALUATION FUNCTION (Metrics + Samples)
# =========================================================
@torch.no_grad()
def evaluate_epoch(model, val_loader, retriever_mgr, tokenizer, device):
    model.eval()
    preds = []
    refs = []
    
    print("\nRunning Validation Generation...")
    # Use tqdm but don't spam newlines
    for batch in tqdm(val_loader, desc="Validating"):
        graphs, _ = batch # Val loader returns (graph, text_emb)
        graphs = graphs.to(device)
        
        # 1. Retrieve Hints (Test mode: exclude_self=False is usually fine for Val if Val != Train)
        # But if Val comes from same distribution, keep True to be safe.
        # Ideally Val set is distinct, so we search in Train Index.
        hint_strings = retriever_mgr.get_hints(graphs, k=TOP_K, exclude_self=False)
        
        hint_inputs = tokenizer(
            hint_strings, padding="longest", truncation=True, max_length=256, return_tensors="pt"
        ).to(device)
        
        # 2. Generate
        gen_ids = model.generate(
            graphs, 
            hint_input_ids=hint_inputs.input_ids,
            hint_mask=hint_inputs.attention_mask
        )
        
        # 3. Decode
        decoded_preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        preds.extend(decoded_preds)
        
        # 4. Store Ground Truth
        for g in graphs.to_data_list():
            refs.append(g.description)

    # Compute Metrics
    print("Computing Metrics...")
    # BLEU-4
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    
    # BERTScore
    P, R, F1 = bert_score_func(
        preds, refs, 
        lang="en", 
        model_type="roberta-base", 
        verbose=False, 
        device=device,
        batch_size=32
    )
    
    return {
        "BLEU-4": bleu.score,
        "BERTScore": F1.mean().item(),
        "Samples": list(zip(preds[:5], refs[:5])) # Return first 5 pairs
    }

# =========================================================
# 4. TRAINING LOOP
# =========================================================
def main():
    print(f"Device: {DEVICE}")
    tokenizer = T5Tokenizer.from_pretrained(MOLT5_MODEL, legacy=False)
    
    from utils.data_utils import load_id2emb 
    train_emb = load_id2emb("embeddings/train_chembed_embeddings.csv")
    val_emb   = load_id2emb("embeddings/validation_chembed_embeddings.csv") # Load Val embeddings too
    
    train_ds = PreprocessedGraphDataset(TRAIN_GRAPHS, train_emb)
    val_ds   = PreprocessedGraphDataset(VAL_GRAPHS, val_emb) # Create Val Dataset
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    # Val loader for metrics
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    # Index loader (Train only)
    index_loader = DataLoader(train_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    retriever_mgr = RetrievalManager(RETRIEVER_PATH, index_loader, DEVICE)
    model_gen = MolGNN_RAG(hidden=128).to(DEVICE)
    optimizer = torch.optim.AdamW(model_gen.parameters(), lr=LR)
    
    print("\n=== Starting RAG Training ===")
    
    best_bert = 0.0
    
    for epoch in range(EPOCHS):
        model_gen.train()
        total_loss = 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in progress:
            graphs, _ = batch 
            graphs = graphs.to(DEVICE)
            
            # A. Retrieve Hints (Exclude self during training!)
            hint_strings = retriever_mgr.get_hints(graphs, k=TOP_K, exclude_self=True)
            
            hint_inputs = tokenizer(
                hint_strings, padding="longest", truncation=True, max_length=256, return_tensors="pt"
            ).to(DEVICE)
            
            targets = [g.description for g in graphs.to_data_list()]
            target_inputs = tokenizer(
                targets, padding="longest", truncation=True, max_length=128, return_tensors="pt"
            ).input_ids.to(DEVICE)
            target_inputs[target_inputs == tokenizer.pad_token_id] = -100 
            
            optimizer.zero_grad()
            loss = model_gen(
                batch_graph=graphs, 
                hint_input_ids=hint_inputs.input_ids,
                hint_mask=hint_inputs.attention_mask,
                labels=target_inputs
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_gen.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress.set_postfix({"loss": loss.item()})
            
        print(f"Epoch {epoch+1} Train Loss: {total_loss / len(train_loader):.4f}")
        
        # --- VALIDATION & SAMPLES ---
        # Run every epoch (or every other epoch if slow)
        metrics = evaluate_epoch(model_gen, val_loader, retriever_mgr, tokenizer, DEVICE)
        
        print(f"\n[Validation Epoch {epoch+1}]")
        print(f"  BLEU-4:    {metrics['BLEU-4']:.2f}")
        print(f"  BERTScore: {metrics['BERTScore']:.4f}")
        print("\n  --- Sample Generations ---")
        for i, (pred, truth) in enumerate(metrics["Samples"]):
            print(f"  Sample {i+1}:")
            print(f"    Pred:  {pred}")
            print(f"    Truth: {truth}\n")
            
        # Save Best
        if metrics['BERTScore'] > best_bert:
            best_bert = metrics['BERTScore']
            torch.save(model_gen.state_dict(), MODEL_PATH)
            print(f"  >>> New Best Model Saved (BERTScore: {best_bert:.4f})")

if __name__ == "__main__":
    main()

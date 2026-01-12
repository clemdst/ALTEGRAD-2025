
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

# PyG Imports
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import TransformerConv

# Import Utils
from utils.data_utils import PreprocessedGraphDataset, collate_fn, load_id2emb, x_map, e_map

# Import just the Retriever Model Class (not the manager)
from strategy.strategy_6.train_dual_tower import MolTransformerDual 

# =========================================================
# CONFIGURATION
# =========================================================
TRAIN_GRAPHS   = "data/train_graphs.pkl"
TEST_GRAPHS    = "data/test_graphs.pkl"
TRAIN_EMB_PATH = "embeddings/train_chembed_embeddings.csv"

RETRIEVER_PATH = "models/model_strategy_6.pt"
GENERATOR_PATH = "models/rag_generator_strategy_8.pt"
SUBMISSION_CSV = "submission_strategy_8.csv"

MOLT5_MODEL = "laituan245/molt5-small"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 32
TOP_K       = 3

# =========================================================
# 1. REDEFINE CLASSES (To ensure fix is applied)
# =========================================================

class MolGNN_RAG(nn.Module):
    # Copy of the class to ensure shape matching
    def __init__(self, hidden=128, layers=3, heads=4, text_model="laituan245/molt5-small"):
        super().__init__()
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
        self.t5 = T5ForConditionalGeneration.from_pretrained(text_model)
        self.graph_proj = nn.Linear(hidden, self.t5.config.d_model)

    def encode_graph(self, batch):
        node_feats = [emb(batch.x[:, i]) for i, emb in enumerate(self.node_emb)]
        x = self.node_proj(torch.cat(node_feats, dim=-1))
        edge_feats = [emb(batch.edge_attr[:, i]) for i, emb in enumerate(self.edge_emb)]
        edge_attr = self.edge_proj(torch.cat(edge_feats, dim=-1))
        for conv, ln in zip(self.convs, self.layer_norms):
            x = F.relu(ln(x + conv(x, batch.edge_index, edge_attr)))
        x_dense, mask = to_dense_batch(x, batch.batch)
        return self.graph_proj(x_dense), mask

    def generate(self, batch_graph, hint_input_ids, hint_mask, max_new_tokens=128):
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

# --- FIX: Updated RetrievalManager with safe 'exclude_self' logic ---
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
                    
                    # --- CRITICAL FIX IS HERE ---
                    # We only access .description if we are actively excluding self
                    if exclude_self:
                        # Safety check: does the graph even have a description? (Test set doesn't)
                        if hasattr(batch_graph[r], 'description'):
                            target_text = batch_graph[r].description
                            if candidate_text == target_text:
                                continue
                    # -----------------------------

                    valid_hints.append(candidate_text)
                    if len(valid_hints) == k: break
                
                prompt = " ".join([f"Hint: {h}" for h in valid_hints])
                hints_batch.append(prompt)
            return hints_batch

# =========================================================
# 2. MAIN INFERENCE
# =========================================================
@torch.no_grad()
def main():
    print(f"Device: {DEVICE}")
    tokenizer = T5Tokenizer.from_pretrained(MOLT5_MODEL, legacy=False)

    # 1. LOAD DATA FOR INDEXING
    print("Loading Training Data (for Retriever Index)...")
    train_emb = load_id2emb(TRAIN_EMB_PATH)
    train_ds = PreprocessedGraphDataset(TRAIN_GRAPHS, train_emb)
    index_loader = DataLoader(train_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # 2. LOAD TEST DATA
    print("Loading Test Data...")
    test_ds = PreprocessedGraphDataset(TEST_GRAPHS, emb_dict=None) 
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 3. INITIALIZE MODELS
    retriever_mgr = RetrievalManager(RETRIEVER_PATH, index_loader, DEVICE)
    
    print(f"Loading Generator from {GENERATOR_PATH}...")
    model_gen = MolGNN_RAG(hidden=128).to(DEVICE)
    
    if os.path.exists(GENERATOR_PATH):
        model_gen.load_state_dict(torch.load(GENERATOR_PATH, map_location=DEVICE))
    else:
        print(f"ERROR: Generator model not found at {GENERATOR_PATH}")
        return
    model_gen.eval()

    # 4. GENERATE PREDICTIONS
    results = []
    print("\n=== Running Inference on Test Set ===")
    
    for batch in tqdm(test_loader, desc="Generating"):
        # Handle tuple vs single object from loader
        if isinstance(batch, tuple):
            batch_graph = batch[0]
        else:
            batch_graph = batch
            
        batch_graph = batch_graph.to(DEVICE)
        ids = [g.id for g in batch_graph.to_data_list()]

        # A. Get Hints (RAG) - exclude_self=False because test data isn't in training index
        hint_strings = retriever_mgr.get_hints(batch_graph, k=TOP_K, exclude_self=False)
        
        hint_inputs = tokenizer(
            hint_strings, padding="longest", truncation=True, max_length=512, return_tensors="pt"
        ).to(DEVICE)

        # B. Generate
        gen_ids = model_gen.generate(
            batch_graph, 
            hint_input_ids=hint_inputs.input_ids,
            hint_mask=hint_inputs.attention_mask,
            max_new_tokens=128
        )
        
        # C. Decode
        captions = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        
        for mol_id, cap in zip(ids, captions):
            results.append({"ID": mol_id, "description": cap})

    # 5. SAVE SUBMISSION
    df = pd.DataFrame(results)
    df.to_csv(SUBMISSION_CSV, index=False)
    print(f"\nSuccess! Submission saved to: {SUBMISSION_CSV}")
    print(df.head())

if __name__ == "__main__":
    main()

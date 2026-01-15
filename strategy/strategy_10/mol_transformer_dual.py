import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import TransformerConv, global_add_pool, global_mean_pool

from utils.data_utils import (
    x_map, e_map
)

def colbert_score(graph_tokens, text_tokens):
    """
    ColBERT-style late interaction scoring using MaxSim.
    For each query token, find max similarity with document tokens, then sum.
    
    Args:
        graph_tokens: [batch_size, num_graph_tokens, dim] - normalized
        text_tokens: [batch_size, num_text_tokens, dim] - normalized
    Returns:
        scores: [batch_size, batch_size] similarity matrix
    """
    batch_size = graph_tokens.size(0)
    scores = torch.zeros(batch_size, batch_size, device=graph_tokens.device)
    
    for i in range(batch_size):
        # For each text query (i), compute score with all graph documents
        text_query = text_tokens[i:i+1]  # [1, num_text_tokens, dim]
        
        # Compute similarity between this query's tokens and all documents' tokens
        # [1, num_text_tokens, dim] @ [batch_size, dim, num_graph_tokens]
        # -> [batch_size, num_text_tokens, num_graph_tokens]
        sim = torch.matmul(text_query, graph_tokens.transpose(1, 2))  # [batch_size, num_text_tokens, num_graph_tokens]
        
        # MaxSim: for each query token, take max over document tokens
        max_sim = sim.max(dim=-1)[0]  # [batch_size, num_text_tokens]
        
        # Sum across query tokens
        scores[:, i] = max_sim.sum(dim=-1)  # [batch_size]
    
    return scores.t()  # [batch_size, batch_size]

class MolTransformerDual(nn.Module):
    def __init__(self, hidden=128, text_dim=768, out_dim=128, layers=3, heads=4, 
                 use_colbert=True, num_text_tokens=32):
        super().__init__()
        
        self.use_colbert = use_colbert
        self.num_text_tokens = num_text_tokens
        self.out_dim = out_dim
        
        # --- TOWER A: GRAPH ENCODER (Transformer) ---
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
        
        if use_colbert:
            # ColBERT: Project node embeddings to token-level representations
            self.graph_token_proj = nn.Sequential(
                nn.Linear(hidden, out_dim),
                nn.LayerNorm(out_dim)
            )
        else:
            # Traditional: Pool and project to single vector
            self.graph_proj = nn.Sequential(
                nn.Linear(hidden, hidden * 2),
                nn.BatchNorm1d(hidden * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden * 2, out_dim)
            )

        # --- TOWER B: TEXT ENCODER (The "Adapter") ---
        if use_colbert:
            # ColBERT: Generate multiple text tokens from single embedding
            self.text_token_generator = nn.Sequential(
                nn.Linear(text_dim, hidden * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden * 2, num_text_tokens * out_dim)
            )
        else:
            # Traditional: Project text embedding to single vector
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, text_dim),
                nn.BatchNorm1d(text_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(text_dim, out_dim)
            )
            # Init text projection close to identity to start stable
            first_layer = self.text_proj[0]
            last_layer = self.text_proj[4]
            if isinstance(first_layer, nn.Linear):
                torch.nn.init.eye_(first_layer.weight[:text_dim, :text_dim] if text_dim == first_layer.weight.shape[0] else first_layer.weight)
            if isinstance(last_layer, nn.Linear) and out_dim == text_dim:
                torch.nn.init.eye_(last_layer.weight[:out_dim, :text_dim] if out_dim <= last_layer.weight.shape[0] else last_layer.weight)

    def forward_graph(self, batch, return_tokens=None):
        """Pass Graph through Tower A"""
        if return_tokens is None:
            return_tokens = self.use_colbert
            
        node_feats = [emb(batch.x[:, i]) for i, emb in enumerate(self.node_emb)]
        x = self.node_proj(torch.cat(node_feats, dim=-1))
        
        edge_feats = [emb(batch.edge_attr[:, i]) for i, emb in enumerate(self.edge_emb)]
        edge_attr = self.edge_proj(torch.cat(edge_feats, dim=-1))
        
        for conv, ln in zip(self.convs, self.layer_norms):
            x = F.relu(ln(x + conv(x, batch.edge_index, edge_attr)))
        
        if return_tokens:
            # ColBERT: Return token-level embeddings for each node
            tokens = self.graph_token_proj(x)  # [num_nodes, out_dim]
            tokens = F.normalize(tokens, p=2, dim=-1)
            
            # Group by batch to get [batch_size, max_nodes, out_dim]
            from torch_geometric.utils import to_dense_batch
            tokens_dense, mask = to_dense_batch(tokens, batch.batch)
            return tokens_dense  # [batch_size, max_nodes, out_dim]
        else:
            # Traditional: Pool to single vector
            g = global_add_pool(x, batch.batch) + global_mean_pool(x, batch.batch)
            return self.graph_proj(g)

    def forward_text(self, text_emb, return_tokens=None):
        """Pass Text through Tower B"""
        if return_tokens is None:
            return_tokens = self.use_colbert
            
        if return_tokens:
            # ColBERT: Generate multiple text tokens
            tokens_flat = self.text_token_generator(text_emb)  # [batch_size, num_tokens * dim]
            tokens = tokens_flat.view(-1, self.num_text_tokens, self.out_dim)  # [batch_size, num_tokens, dim]
            tokens = F.normalize(tokens, p=2, dim=-1)
            return tokens
        else:
            # Traditional: Single vector
            return self.text_proj(text_emb)

    def forward(self, batch, text_emb):
        """Training Step: Return both vectors or tokens"""
        if self.use_colbert:
            # Return token-level embeddings
            g_tokens = self.forward_graph(batch, return_tokens=True)
            t_tokens = self.forward_text(text_emb, return_tokens=True)
            return g_tokens, t_tokens
        else:
            # Return single vectors
            g_vec = F.normalize(self.forward_graph(batch, return_tokens=False), p=2, dim=-1)
            t_vec = F.normalize(self.forward_text(text_emb, return_tokens=False), p=2, dim=-1)
            return g_vec, t_vec
    
    def compute_similarity(self, batch, text_emb):
        """Compute similarity scores using appropriate method"""
        if self.use_colbert:
            g_tokens, t_tokens = self.forward(batch, text_emb)
            return colbert_score(g_tokens, t_tokens)
        else:
            g_vec, t_vec = self.forward(batch, text_emb)
            return g_vec @ t_vec.T
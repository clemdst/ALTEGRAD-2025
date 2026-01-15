import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import TransformerConv, global_add_pool, global_mean_pool

from utils.data_utils import (
    x_map, e_map
)

def colbert_score(text_tokens, graph_tokens, graph_mask=None):
    """
    ColBERT-style late interaction scoring using MaxSim (Optimized & Vectorized).
    For each query token, find max similarity with document tokens, then sum.
    
    Args:
        text_tokens: [batch_size, num_text_tokens, dim] - normalized (queries)
        graph_tokens: [batch_size, num_graph_tokens, dim] - normalized (documents)
        graph_mask: [batch_size, num_graph_tokens] - Optional mask for padding (True = valid)
    Returns:
        scores: [batch_size, batch_size] similarity matrix (text x graph)
    """
    # Vectorized computation using einsum
    # i = batch_text, j = batch_graph, t = num_text_tokens, g = num_graph_tokens, d = dim
    # text_tokens: [i, t, d], graph_tokens: [j, g, d]
    # Result: [i, j, t, g] = similarity between each text token and each graph token
    scores = torch.einsum('itd,jgd->ijtg', text_tokens, graph_tokens)
    # Result: [batch_text, batch_graph, num_text_tokens, num_graph_tokens]
    
    # Apply mask to graph tokens if provided (mask padding)
    if graph_mask is not None:
        # Expand mask: [batch_graph, num_graph_tokens] -> [1, batch_graph, 1, num_graph_tokens]
        mask_expanded = graph_mask.unsqueeze(0).unsqueeze(2)
        scores = scores.masked_fill(~mask_expanded, float('-inf'))
    
    # MaxSim: for each text token, take max similarity over all graph tokens
    max_scores = scores.max(dim=-1)[0]  # [batch_text, batch_graph, num_text_tokens]
    
    # Sum across text tokens to get final scores
    final_scores = max_scores.sum(dim=-1)  # [batch_text, batch_graph]
    
    return final_scores

class MolTransformerDual(nn.Module):
    def __init__(self, hidden=128, text_dim=768, out_dim=192, layers=3, heads=4, 
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
            # ColBERT: Support both real ColBERT tokens and generated tokens
            # If text_dim is already expanded (num_tokens * hidden), we have real tokens
            # Otherwise, generate pseudo-tokens from mean-pooled embedding
            if text_dim == num_text_tokens * 768:  # Real ColBERT tokens (flattened)
                # Project each real token: [batch, num_tokens, 768] -> [batch, num_tokens, out_dim]
                self.text_token_projector = nn.Sequential(
                    nn.Linear(768, hidden),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden, out_dim),
                    nn.LayerNorm(out_dim)
                )
                self.using_real_tokens = True
            else:
                # Generate pseudo-tokens from mean-pooled embedding (fallback)
                self.text_token_generator = nn.Sequential(
                    nn.Linear(text_dim, hidden * 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden * 2, num_text_tokens * out_dim)
                )
                self.using_real_tokens = False
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
            return tokens_dense, mask  # Return mask for padding handling
        else:
            # Traditional: Pool to single vector
            g = global_add_pool(x, batch.batch) + global_mean_pool(x, batch.batch)
            return self.graph_proj(g)

    def forward_text(self, text_emb, return_tokens=None):
        """Pass Text through Tower B"""
        if return_tokens is None:
            return_tokens = self.use_colbert
            
        if return_tokens:
            # Check if we have real ColBERT tokens or need to generate pseudo-tokens
            if text_emb.dim() == 3:
                # Real ColBERT tokens: [batch_size, num_tokens, 768]
                # Project each token independently
                batch_size, num_tokens, token_dim = text_emb.shape
                # Flatten for projection: [batch * num_tokens, token_dim]
                tokens_flat = text_emb.reshape(-1, token_dim)
                projected = self.text_token_projector(tokens_flat)
                # Reshape back: [batch_size, num_tokens, out_dim]
                tokens = projected.reshape(batch_size, num_tokens, self.out_dim)
            elif hasattr(self, 'using_real_tokens') and self.using_real_tokens:
                # Real tokens but flattened: [batch_size, num_tokens * 768]
                batch_size = text_emb.size(0)
                # Reshape to [batch_size, num_tokens, 768]
                text_tokens = text_emb.view(batch_size, self.num_text_tokens, 768)
                # Project each token
                tokens_flat = text_tokens.reshape(-1, 768)
                projected = self.text_token_projector(tokens_flat)
                tokens = projected.reshape(batch_size, self.num_text_tokens, self.out_dim)
            else:
                # Generate pseudo-tokens from mean-pooled embedding (fallback)
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
            g_tokens, g_mask = self.forward_graph(batch, return_tokens=True)
            t_tokens = self.forward_text(text_emb, return_tokens=True)
            return g_tokens, t_tokens, g_mask
        else:
            # Return single vectors
            g_vec = F.normalize(self.forward_graph(batch, return_tokens=False), p=2, dim=-1)
            t_vec = F.normalize(self.forward_text(text_emb, return_tokens=False), p=2, dim=-1)
            return g_vec, t_vec
    
    def compute_similarity(self, batch, text_emb):
        """Compute similarity scores using appropriate method"""
        if self.use_colbert:
            g_tokens, t_tokens, g_mask = self.forward(batch, text_emb)
            return colbert_score(t_tokens, g_tokens, g_mask)  # ✅ Correct order: text first
        else:
            g_vec, t_vec = self.forward(batch, text_emb)
            return g_vec @ t_vec.T
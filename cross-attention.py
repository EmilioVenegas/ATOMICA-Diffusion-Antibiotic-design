"""
cross_attention.py
SE(3)-equivariant cross-attention between ligand and ATOMICA embeddings
"""

import torch
import torch.nn as nn


class SE3EquivariantCrossAttention(nn.Module):
    """
    Cross-attention module that maintains SE(3)-equivariance.
    Ligand nodes attend to ATOMICA pocket embeddings.
    """
    def __init__(self, hidden_nf, atomica_dim, num_heads=4, edge_cutoff=10.0):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.atomica_dim = atomica_dim
        self.num_heads = num_heads
        self.head_dim = hidden_nf // num_heads
        self.edge_cutoff = edge_cutoff
        
        assert hidden_nf % num_heads == 0
        
        # Attention projections
        self.q_proj = nn.Linear(hidden_nf, hidden_nf)
        self.k_proj = nn.Linear(atomica_dim, hidden_nf)
        self.v_proj = nn.Linear(atomica_dim, hidden_nf)
        self.out_proj = nn.Linear(hidden_nf, hidden_nf)
        
        # Distance-aware edge features
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, num_heads)
        )
        
        # Equivariant coordinate update
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, 1)
        )
        
    def forward(self, h_lig, x_lig, h_atomica, x_pocket):
        """
        Args:
            h_lig: [N_lig, hidden_nf] - ligand features
            x_lig: [N_lig, 3] - ligand coordinates
            h_atomica: [N_pocket, atomica_dim] - ATOMICA embeddings
            x_pocket: [N_pocket, 3] - pocket coordinates
            
        Returns:
            h_out: [N_lig, hidden_nf] - updated features
            x_out: [N_lig, 3] - updated coordinates
        """
        N_lig = h_lig.shape[0]
        N_pocket = h_atomica.shape[0]
        
        # Compute relative positions and distances
        rel_pos = x_pocket.unsqueeze(0) - x_lig.unsqueeze(1)  # [N_lig, N_pocket, 3]
        distances = torch.norm(rel_pos, dim=-1, keepdim=True)  # [N_lig, N_pocket, 1]
        
        # Distance-based modulation
        edge_attr = self.edge_mlp(distances)  # [N_lig, N_pocket, num_heads]
        
        # Distance cutoff mask
        mask = (distances.squeeze(-1) < self.edge_cutoff).float()
        
        # Multi-head attention
        Q = self.q_proj(h_lig).view(N_lig, self.num_heads, self.head_dim)
        K = self.k_proj(h_atomica).view(N_pocket, self.num_heads, self.head_dim)
        V = self.v_proj(h_atomica).view(N_pocket, self.num_heads, self.head_dim)
        
        # Attention scores
        scores = torch.einsum('ihd,jhd->ijh', Q, K) / (self.head_dim ** 0.5)
        scores = scores + edge_attr
        
        # Apply mask and softmax
        scores = scores.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        attn_weights = torch.softmax(scores, dim=1)
        attn_weights = torch.nan_to_num(attn_weights, 0.0)
        
        # Attend to values (scalar features - invariant)
        h_attended = torch.einsum('ijh,jhd->ihd', attn_weights, V)
        h_attended = h_attended.reshape(N_lig, -1)
        h_out = h_lig + self.out_proj(h_attended)
        
        # Equivariant coordinate update
        normalized_rel_pos = rel_pos / (distances + 1e-8)
        avg_attn = attn_weights.mean(dim=-1, keepdim=True)
        coord_messages = (avg_attn * normalized_rel_pos).sum(dim=1)
        coord_scale = self.coord_mlp(h_attended)
        x_out = x_lig + coord_scale * coord_messages
        
        return h_out, x_out
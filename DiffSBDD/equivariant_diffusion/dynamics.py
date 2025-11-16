import torch
import torch.nn as nn
from equivariant_diffusion.egnn_new import EGNN, SE3CrossAttention
import numpy as np



class AtomicaDynamics(nn.Module):
    def __init__(self, atom_nf, context_nf, n_dims, hidden_nf, device, act_fn, n_layers, attention,
                 tanh, norm_constant, inv_sublayers, sin_embedding, normalization_factor,
                 aggregation_method, edge_cutoff_ligand, edge_cutoff_interaction,
                 reflection_equivariant, edge_embedding_dim, **kwargs):
        super().__init__()
        self.update_pocket_coords = False
        self.edge_cutoff_l = edge_cutoff_ligand
        self.edge_cutoff_i = edge_cutoff_interaction
        self.condition_time = kwargs.get('condition_time', True)
        self.edge_nf = edge_embedding_dim
        self.n_dims = n_dims
        
        # A single, stable embedding/encoding layer for each input type.
        self.atom_encoder = nn.Linear(atom_nf, hidden_nf)
        self.context_encoder = nn.Linear(context_nf, hidden_nf)
        
        # A single, stable decoding layer.
        self.atom_decoder = nn.Linear(hidden_nf, atom_nf)
        
        self.edge_embedding = nn.Embedding(2, self.edge_nf) if self.edge_nf is not None else None
        
        dynamics_node_nf = hidden_nf + 1 if self.condition_time else hidden_nf
        
        # The "pure" processing blocks that receive ALREADY-EMBEDDED features.
        self.egnn = EGNN(
            in_node_nf=dynamics_node_nf, in_edge_nf=self.edge_nf, hidden_nf=hidden_nf, device=device, act_fn=act_fn,
            n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
            inv_sublayers=inv_sublayers, sin_embedding=sin_embedding, normalization_factor=normalization_factor,
            aggregation_method=aggregation_method, reflection_equiv=reflection_equivariant
        )
        
        self.cross_attention = SE3CrossAttention(
            in_node_nf_q=dynamics_node_nf, in_node_nf_kv=dynamics_node_nf, in_edge_nf=self.edge_nf,
            hidden_nf=hidden_nf, device=device, act_fn=act_fn, n_layers=n_layers, attention=attention, tanh=tanh,
            norm_constant=norm_constant, inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
            normalization_factor=normalization_factor, aggregation_method=aggregation_method,
            reflection_equiv=reflection_equivariant
        )

    def forward(self, xh_lig, xh_context, t, mask_lig, mask_context):
        x_l, h_l = xh_lig[:, :self.n_dims], xh_lig[:, self.n_dims:]
        x_p, h_p = xh_context[:, :self.n_dims], xh_context[:, self.n_dims:]

        h_l_emb = self.atom_encoder(h_l)
        h_p_emb = self.context_encoder(h_p)

        if self.condition_time:
            time_emb = t[mask_lig] if t.numel() > 1 else t.expand(h_l_emb.size(0), 1)
            h_l_t = torch.cat([h_l_emb, time_emb], dim=1)
            time_emb_p = t[mask_context] if t.numel() > 1 else t.expand(h_p_emb.size(0), 1)
            h_p_t = torch.cat([h_p_emb, time_emb_p], dim=1)
        else:
            h_l_t, h_p_t = h_l_emb, h_p_emb

        edges_ll = self.get_ligand_edges(mask_lig, x_l)
        edge_attr_ll = self.edge_embedding(torch.ones(edges_ll.size(1), device=edges_ll.device, dtype=torch.long)) if self.edge_nf > 0 else None
        h_intermediate, x_intermediate = self.egnn(h_l_t, x_l, edges_ll, edge_attr=edge_attr_ll, batch_mask=mask_lig)
        
        edges_lp = self.get_cross_edges(mask_lig, mask_context, x_intermediate, x_p)
        edge_attr_lp = self.edge_embedding(torch.zeros(edges_lp.size(1), device=edges_lp.device, dtype=torch.long)) if self.edge_nf > 0 else None
        h_final_emb, x_final = self.cross_attention(h_intermediate, x_intermediate, h_p_t, x_p, edges_lp, edge_attr=edge_attr_lp, batch_mask=mask_lig)
        
        final_velocity = x_final - x_l
        h_features_final = h_final_emb[:, :-1] if self.condition_time else h_final_emb
        final_features_update = self.atom_decoder(h_features_final)

        ligand_update = torch.cat([final_velocity, final_features_update], dim=-1)
        pocket_update = torch.zeros_like(xh_context)

        return ligand_update, pocket_update

    def get_ligand_edges(self, mask, x):
        adj = mask[:, None] == mask[None, :]
        if self.edge_cutoff_l is not None:
            adj = adj & (torch.cdist(x, x) <= self.edge_cutoff_l)
        torch.diagonal(adj).fill_(False)
        return torch.stack(torch.where(adj))

    def get_cross_edges(self, mask_l, mask_p, x_l, x_p):
        if len(x_p) == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=x_l.device)
        adj = mask_l[:, None] == mask_p[None, :]
        if self.edge_cutoff_i is not None:
            adj = adj & (torch.cdist(x_l, x_p) <= self.edge_cutoff_i)
        return torch.stack(torch.where(adj))
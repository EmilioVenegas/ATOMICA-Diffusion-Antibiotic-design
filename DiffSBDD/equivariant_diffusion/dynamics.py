import torch
import torch.nn as nn
from equivariant_diffusion.egnn_new import EGNN, SE3CrossAttention
import numpy as np


class EGNNDynamics(nn.Module):
    def __init__(self, atom_nf, residue_nf,
                 n_dims, joint_nf=16, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics',
                 norm_constant=0, inv_sublayers=2, sin_embedding=False,
                 normalization_factor=100, aggregation_method='sum',
                 update_pocket_coords=True, edge_cutoff_ligand=None,
                 edge_cutoff_pocket=None, edge_cutoff_interaction=None,
                 reflection_equivariant=True, edge_embedding_dim=None):
        super().__init__()
        # This is the original, non-ATOMICA dynamics class. It is kept for completeness.
        # All changes are in AtomicaDynamics.
        self.mode = mode
        self.edge_cutoff_l = edge_cutoff_ligand
        self.edge_cutoff_p = edge_cutoff_pocket
        self.edge_cutoff_i = edge_cutoff_interaction
        self.edge_nf = edge_embedding_dim
        self.atom_encoder = nn.Sequential(nn.Linear(atom_nf, 2 * atom_nf), act_fn, nn.Linear(2 * atom_nf, joint_nf))
        self.atom_decoder = nn.Sequential(nn.Linear(joint_nf, 2 * atom_nf), act_fn, nn.Linear(2 * atom_nf, atom_nf))
        self.residue_encoder = nn.Sequential(nn.Linear(residue_nf, 2 * residue_nf), act_fn, nn.Linear(2 * residue_nf, joint_nf))
        self.residue_decoder = nn.Sequential(nn.Linear(joint_nf, 2 * residue_nf), act_fn, nn.Linear(2 * residue_nf, residue_nf))
        self.edge_embedding = nn.Embedding(3, self.edge_nf) if self.edge_nf is not None else None
        self.edge_nf = 0 if self.edge_nf is None else self.edge_nf
        dynamics_node_nf = joint_nf + 1 if condition_time else joint_nf
        self.egnn = EGNN(
            in_node_nf=dynamics_node_nf, in_edge_nf=self.edge_nf, hidden_nf=hidden_nf, device=device, act_fn=act_fn,
            n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
            inv_sublayers=inv_sublayers, sin_embedding=sin_embedding, normalization_factor=normalization_factor,
            aggregation_method=aggregation_method, reflection_equiv=reflection_equivariant
        )
        self.node_nf = dynamics_node_nf
        self.update_pocket_coords = update_pocket_coords
        self.device = device
        self.n_dims = n_dims
        self.condition_time = condition_time

    def forward(self, xh_lig, xh_context, t, mask_lig, mask_context):
        h_lig = self.atom_encoder(xh_lig[:, self.n_dims:])
        h_pocket = self.residue_encoder(xh_context[:, self.n_dims:])
        if self.condition_time:
            if t.numel() == 1:
                h_time = torch.empty_like(h_lig[:, 0:1]).fill_(t.item())
            else:
                h_time = t[torch.cat((mask_lig, mask_context))]
            h = torch.cat([torch.cat((h_lig, h_pocket)), h_time], dim=1)
        else:
            h = torch.cat((h_lig, h_pocket))
        x = torch.cat((xh_lig[:, :self.n_dims], xh_context[:, :self.n_dims]))
        mask = torch.cat((mask_lig, mask_context))
        edges = self.get_edges(mask_lig, mask_context, xh_lig[:, :self.n_dims], xh_context[:, :self.n_dims])
        edge_attr = self.get_edge_attr(edges, mask_lig, mask_context) if self.edge_nf > 0 else None
        update_coords_mask = torch.cat([torch.ones_like(mask_lig), torch.zeros_like(mask_context) if not self.update_pocket_coords else torch.ones_like(mask_context)]).unsqueeze(1)
        h_final, x_final = self.egnn(h, x, edges, update_coords_mask=update_coords_mask, edge_attr=edge_attr, batch_mask=mask)
        vel = (x_final - x)
        vel_lig, vel_pocket = vel[:len(mask_lig)], vel[len(mask_lig):]
        h_final_lig, h_final_pocket = h_final[:len(mask_lig)], h_final[len(mask_lig):]
        if self.condition_time:
            h_final_lig, h_final_pocket = h_final_lig[:, :-1], h_final_pocket[:, :-1]
        out_h_lig = self.atom_decoder(h_final_lig)
        out_h_pocket = self.residue_decoder(h_final_pocket)
        return torch.cat([vel_lig, out_h_lig], dim=1), torch.cat([vel_pocket, out_h_pocket], dim=1)

    def get_edges(self, bl, bp, xl, xp):
        adj_l = bl[:, None] == bl[None, :]
        adj_p = bp[:, None] == bp[None, :]
        adj_c = bl[:, None] == bp[None, :]
        if self.edge_cutoff_l is not None: adj_l = adj_l & (torch.cdist(xl, xl) <= self.edge_cutoff_l)
        if self.edge_cutoff_p is not None: adj_p = adj_p & (torch.cdist(xp, xp) <= self.edge_cutoff_p)
        if self.edge_cutoff_i is not None: adj_c = adj_c & (torch.cdist(xl, xp) <= self.edge_cutoff_i)
        adj = torch.cat([torch.cat([adj_l, adj_c], 1), torch.cat([adj_c.T, adj_p], 1)], 0)
        return torch.stack(torch.where(adj))

    def get_edge_attr(self, edges, ml, mp):
        len_l = len(ml)
        edge_type = torch.zeros(edges.size(1), device=edges.device, dtype=torch.long)
        edge_type[(edges[0] < len_l) & (edges[1] < len_l)] = 1
        edge_type[(edges[0] >= len_l) & (edges[1] >= len_l)] = 2
        return self.edge_embedding(edge_type)


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
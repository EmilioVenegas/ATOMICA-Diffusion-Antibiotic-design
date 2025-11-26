import torch
import torch.nn as nn
from equivariant_diffusion.egnn_new import EquivariantBlock, CrossEquivariantBlock, SinusoidsEmbeddingNew, InteractionBlock

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
        self.n_layers = n_layers
        self.device = device
        self.coords_range = kwargs.get('coords_range', 10.0)
        
        # A single, stable embedding/encoding layer for each input type.
        self.atom_encoder = nn.Linear(atom_nf, hidden_nf)
        self.context_encoder = nn.Linear(context_nf, hidden_nf)
        
        # A single, stable decoding layer.
        self.atom_decoder = nn.Linear(hidden_nf, atom_nf)
        
        self.edge_embedding = nn.Embedding(2, self.edge_nf) if self.edge_nf is not None else None
        
        # --- FIX: Stronger Time Embedding ---
        if self.condition_time:
            self.time_embedding = SinusoidsEmbeddingNew()
            self.time_emb_dim = self.time_embedding.dim
        else:
            self.time_embedding = None
            self.time_emb_dim = 0
            
        dynamics_node_nf = hidden_nf + self.time_emb_dim if self.condition_time else hidden_nf
        # --- END FIX ---
        
        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim
        else:
            self.sin_embedding = None
            edge_feat_nf = 1

        if self.edge_nf is not None:
             edge_feat_nf += self.edge_nf

        # --- INTERACTION BLOCK ARCHITECTURE ---
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                InteractionBlock(
                    hidden_nf=dynamics_node_nf,
                    edge_feat_nf=edge_feat_nf,
                    device=device,
                    act_fn=act_fn,
                    n_layers=inv_sublayers,
                    attention=attention,
                    norm_diff=True,
                    tanh=tanh,
                    coords_range=self.coords_range,
                    norm_constant=norm_constant,
                    sin_embedding=self.sin_embedding,
                    normalization_factor=normalization_factor,
                    aggregation_method=aggregation_method,
                    reflection_equiv=reflection_equivariant
                )
            )

    def forward(self, xh_lig, xh_context, t, mask_lig, mask_context):
        x_l, h_l = xh_lig[:, :self.n_dims], xh_lig[:, self.n_dims:]
        x_p, h_p = xh_context[:, :self.n_dims], xh_context[:, self.n_dims:]

        h_l_emb = self.atom_encoder(h_l)
        h_p_emb = self.context_encoder(h_p)

        if self.condition_time:
            # --- FIX: Apply Sinusoidal Embedding ---
            if t.numel() > 1:
                t_emb = self.time_embedding(t)
                time_emb = t_emb[mask_lig]
                time_emb_p = t_emb[mask_context]
            else:
                t_emb = self.time_embedding(t).view(1, -1)
                time_emb = t_emb.expand(h_l_emb.size(0), -1)
                time_emb_p = t_emb.expand(h_p_emb.size(0), -1)
            
            h_l_t = torch.cat([h_l_emb, time_emb], dim=1)
            h_p_t = torch.cat([h_p_emb, time_emb_p], dim=1)
            # --- END FIX ---
        else:
            h_l_t, h_p_t = h_l_emb, h_p_emb

        # --- Iterative Updates ---
        for i, layer in enumerate(self.layers):
            # 1. Edges
            edges_ll = self.get_ligand_edges(mask_lig, x_l)
            edge_attr_ll = self.edge_embedding(torch.ones(edges_ll.size(1), device=edges_ll.device, dtype=torch.long)) if self.edge_nf > 0 else None
            
            edges_lp = self.get_cross_edges(mask_lig, mask_context, x_l, x_p)
            edge_attr_lp = self.edge_embedding(torch.zeros(edges_lp.size(1), device=edges_lp.device, dtype=torch.long)) if self.edge_nf > 0 else None
            
            h_l_t, x_l = layer(h_l_t, x_l, h_p_t, x_p, edges_ll, edges_lp, 
                               node_mask_l=None, edge_mask_ll=None, edge_mask_lp=None,
                               edge_attr_ll=edge_attr_ll, edge_attr_lp=edge_attr_lp, batch_mask_l=mask_lig)
        
        final_velocity = x_l - xh_lig[:, :self.n_dims] # Calculate displacement
        
        # --- FIX: Correctly strip multi-dimensional time embedding ---
        if self.condition_time:
            h_features_final = h_l_t[:, :-self.time_emb_dim]
        else:
            h_features_final = h_l_t
        # --- END FIX ---
            
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
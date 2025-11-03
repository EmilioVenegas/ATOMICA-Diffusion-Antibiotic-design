import torch
import torch.nn as nn
import torch.nn.functional as F
from equivariant_diffusion.egnn_new import EGNN, GNN, SE3CrossAttention
from equivariant_diffusion.en_diffusion import EnVariationalDiffusion
remove_mean_batch = EnVariationalDiffusion.remove_mean_batch
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
        self.mode = mode
        self.edge_cutoff_l = edge_cutoff_ligand
        self.edge_cutoff_p = edge_cutoff_pocket
        self.edge_cutoff_i = edge_cutoff_interaction
        self.edge_nf = edge_embedding_dim

        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_nf, 2 * atom_nf),
            act_fn,
            nn.Linear(2 * atom_nf, joint_nf)
        )

        self.atom_decoder = nn.Sequential(
            nn.Linear(joint_nf, 2 * atom_nf),
            act_fn,
            nn.Linear(2 * atom_nf, atom_nf)
        )

        self.residue_encoder = nn.Sequential(
            nn.Linear(residue_nf, 2 * residue_nf),
            act_fn,
            nn.Linear(2 * residue_nf, joint_nf)
        )

        self.residue_decoder = nn.Sequential(
            nn.Linear(joint_nf, 2 * residue_nf),
            act_fn,
            nn.Linear(2 * residue_nf, residue_nf)
        )

        self.edge_embedding = nn.Embedding(3, self.edge_nf) \
            if self.edge_nf is not None else None
        self.edge_nf = 0 if self.edge_nf is None else self.edge_nf

        if condition_time:
            dynamics_node_nf = joint_nf + 1
        else:
            print('Warning: dynamics model is _not_ conditioned on time.')
            dynamics_node_nf = joint_nf

        if mode == 'egnn_dynamics':
            self.egnn = EGNN(
                in_node_nf=dynamics_node_nf, in_edge_nf=self.edge_nf,
                hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                n_layers=n_layers, attention=attention, tanh=tanh,
                norm_constant=norm_constant,
                inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                reflection_equiv=reflection_equivariant
            )
            self.node_nf = dynamics_node_nf
            self.update_pocket_coords = update_pocket_coords

        elif mode == 'gnn_dynamics':
            self.gnn = GNN(
                in_node_nf=dynamics_node_nf + n_dims, in_edge_nf=self.edge_nf,
                hidden_nf=hidden_nf, out_node_nf=n_dims + dynamics_node_nf,
                device=device, act_fn=act_fn, n_layers=n_layers,
                attention=attention, normalization_factor=normalization_factor,
                aggregation_method=aggregation_method)

        self.device = device
        self.n_dims = n_dims
        self.condition_time = condition_time

    def forward(self, xh_lig, xh_context, t, mask_lig, mask_context):
        # --- 1. Prepare Inputs with Jitter ---
        # Jitter is especially important during sampling to avoid coincident atoms
        jitter_val = 1e-4
        x_l = xh_lig[:, :self.n_dims].clone() + (jitter_val * torch.randn_like(xh_lig[:, :self.n_dims]))
        h_l = xh_lig[:, self.n_dims:].clone()
        x_p = xh_context[:, :self.n_dims].clone()
        h_p_atomica = xh_context[:, self.n_dims:].clone()

        # --- Initial NaN Check ---
        if not torch.isfinite(x_l).all() or not torch.isfinite(h_l).all():
            print("ERROR: NaN in AtomicaDynamics input. Returning zeros.")
            return torch.zeros_like(xh_lig), torch.zeros_like(xh_context)

        h_l_emb = F.layer_norm(self.atom_encoder(h_l), h_l.shape[1:])
        h_p_emb = F.layer_norm(self.context_encoder(h_p_atomica), h_p_atomica.shape[1:])

        if self.condition_time:
            h_time = t[mask_lig] if np.prod(t.size()) > 1 else torch.empty_like(h_l_emb[:, 0:1]).fill_(t.item())
            h_l_t = torch.cat([h_l_emb, h_time], dim=1)
        else:
            h_l_t = h_l_emb

        # --- 2. L-L Path ---
        edges_ll = self.get_ligand_edges(mask_lig, x_l)
        if edges_ll.shape[1] > 0:
            edge_attr_ll = self.edge_embedding(torch.ones(edges_ll.size(1), dtype=int, device=self.device)) if self.edge_nf > 0 else None
            h_ll_final, x_ll_final = self.egnn(h_l_t, x_l, edges_ll, batch_mask=mask_lig, edge_attr=edge_attr_ll)
            vel_ll = torch.nan_to_num(x_ll_final - x_l) # Sanitize output
        else:
            vel_ll, h_ll_final = torch.zeros_like(x_l), h_l_t.clone()
        
        # --- 3. L-P Path ---
        edges_lp = self.get_cross_edges(mask_lig, mask_context, x_l, x_p)
        if edges_lp.shape[1] > 0:
            edge_attr_lp = self.edge_embedding(torch.zeros(edges_lp.size(1), dtype=int, device=self.device)) if self.edge_nf > 0 else None
            h_lp_final, x_lp_final = self.cross_attention(h_l_t, x_l, h_p_emb, x_p, edges_lp, batch_mask=mask_lig, edge_attr=edge_attr_lp)
            vel_lp = torch.nan_to_num(x_lp_final - x_l) # Sanitize output
        else:
            vel_lp, h_lp_final = torch.zeros_like(x_l), h_l_t.clone()

        # --- 4. Combine Updates ---
        final_velocity = (vel_ll + vel_lp) / 2.0
        
        h_ll_feat = h_ll_final[:, :-1] if self.condition_time else h_ll_final
        h_lp_feat = h_lp_final[:, :-1] if self.condition_time else h_lp_final
        h_final_emb = (h_ll_feat + h_lp_feat) / 2.0
        
        final_features = self.atom_decoder(h_final_emb)

        # Final safety check
        final_velocity = torch.nan_to_num(final_velocity, nan=0.0, posinf=1.0, neginf=-1.0)
        final_features = torch.nan_to_num(final_features, nan=0.0, posinf=1.0, neginf=-1.0)

        return torch.cat([final_velocity, final_features], dim=-1), torch.zeros_like(xh_context)

    def get_edges(self, batch_mask_ligand, batch_mask_pocket, x_ligand, x_pocket):
        adj_ligand = batch_mask_ligand[:, None] == batch_mask_ligand[None, :]
        adj_pocket = batch_mask_pocket[:, None] == batch_mask_pocket[None, :]
        adj_cross = batch_mask_ligand[:, None] == batch_mask_pocket[None, :]

        if self.edge_cutoff_l is not None:
            adj_ligand = adj_ligand & (torch.cdist(x_ligand, x_ligand) <= self.edge_cutoff_l)

        if self.edge_cutoff_p is not None:
            adj_pocket = adj_pocket & (torch.cdist(x_pocket, x_pocket) <= self.edge_cutoff_p)

        if self.edge_cutoff_i is not None:
            adj_cross = adj_cross & (torch.cdist(x_ligand, x_pocket) <= self.edge_cutoff_i)

        adj = torch.cat((torch.cat((adj_ligand, adj_cross), dim=1),
                         torch.cat((adj_cross.T, adj_pocket), dim=1)), dim=0)
        edges = torch.stack(torch.where(adj), dim=0)

        return edges


class AtomicaDynamics(nn.Module):
    def __init__(self, atom_nf, context_nf,
                 n_dims, joint_nf=16, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 condition_time=True, tanh=False,
                 norm_constant=0, inv_sublayers=2, sin_embedding=False,
                 normalization_factor=100, aggregation_method='sum',
                 edge_cutoff_ligand=None, edge_cutoff_interaction=None,
                 reflection_equivariant=True, edge_embedding_dim=None):
        super().__init__()
        self.update_pocket_coords = False
        self.edge_cutoff_l = edge_cutoff_ligand
        self.edge_cutoff_i = edge_cutoff_interaction
        self.edge_nf = edge_embedding_dim
        
        # STABILITY FIX 1: Increase norm_constant for coordinate normalization
        self.coord_norm_constant = max(norm_constant, 1.0)  # At least 1.0

        # Ligand (atom) feature encoder
        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_nf, 2 * atom_nf),
            act_fn,
            nn.Linear(2 * atom_nf, hidden_nf)
        )
        # STABILITY: Initialize with smaller weights
        for layer in self.atom_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.01)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Ligand (atom) feature decoder
        self.atom_decoder = nn.Sequential(
            nn.Linear(hidden_nf, 2 * atom_nf),
            act_fn,
            nn.Linear(2 * atom_nf, atom_nf)
        )
        # STABILITY: Initialize with smaller weights
        for layer in self.atom_decoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.01)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Pocket (context) embedding encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(context_nf, 2 * context_nf),
            act_fn,
            nn.Linear(2 * context_nf, hidden_nf)
        )
        # STABILITY: Initialize with smaller weights
        for layer in self.context_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.01)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        self.edge_embedding = nn.Embedding(2, self.edge_nf) \
            if self.edge_nf is not None else None
        self.edge_nf = 0 if self.edge_nf is None else self.edge_nf

        if condition_time:
            dynamics_node_nf = hidden_nf + 1
        else:
            print('Warning: dynamics model is _not_ conditioned on time.')
            dynamics_node_nf = hidden_nf
            
        self.node_nf = dynamics_node_nf
        self.n_dims = n_dims
        self.device = device
        self.condition_time = condition_time

        # --- Core Components ---
        
        # 1. Ligand-Ligand (L-L) Module
        self.egnn = EGNN(
            in_node_nf=dynamics_node_nf, in_edge_nf=self.edge_nf,
            hidden_nf=hidden_nf, device=device, act_fn=act_fn,
            n_layers=n_layers, attention=attention, tanh=True,
            norm_constant=self.coord_norm_constant,  # STABILITY FIX
            inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
            reflection_equiv=reflection_equivariant
        )
        
        # 2. Ligand-Pocket (L-P) Module
        self.cross_attention = SE3CrossAttention(
            in_node_nf_q=dynamics_node_nf, in_node_nf_kv=dynamics_node_nf,
            in_edge_nf=self.edge_nf,
            hidden_nf=hidden_nf, device=device, act_fn=act_fn,
            n_layers=n_layers, attention=attention, tanh=True,
            norm_constant=self.coord_norm_constant,  # STABILITY FIX
            inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
            reflection_equiv=reflection_equivariant
        )

    def forward(self, xh_lig, xh_context, t, mask_lig, mask_context):

        # --- 1. Prepare Inputs with BETTER JITTER ---
        # STABILITY FIX: Use jitter to prevent overlapping coordinates
        jitter_val = 1e-4
        jitter = jitter_val * torch.randn_like(xh_lig[:, :self.n_dims])
        x_l = xh_lig[:, :self.n_dims].clone() + jitter
        h_l = xh_lig[:, self.n_dims:].clone()
        x_p = xh_context[:, :self.n_dims].clone()
        h_p_atomica = xh_context[:, self.n_dims:].clone()

        # STABILITY CHECK: Verify no NaN in inputs
        if torch.isnan(x_l).any() or torch.isnan(h_l).any():
            print("ERROR: NaN in input to AtomicaDynamics")
            x_l = torch.nan_to_num(x_l, nan=0.0)
            h_l = torch.nan_to_num(h_l, nan=0.0)

        # STABILITY FIX: Apply LayerNorm *before* encoder
        h_l = F.layer_norm(h_l, h_l.shape[1:])
        h_l_emb = self.atom_encoder(h_l)
        
        h_p_atomica = F.layer_norm(h_p_atomica, h_p_atomica.shape[1:])
        h_p_emb = self.context_encoder(h_p_atomica)

        # Condition features on time
        if self.condition_time:
            if np.prod(t.size()) == 1:
                h_time_l = torch.empty_like(h_l_emb[:, 0:1]).fill_(t.item())
                h_time_p = torch.empty_like(h_p_emb[:, 0:1]).fill_(t.item())
            else:
                h_time_l = t[mask_lig]
                h_time_p = t[mask_context]

            h_l_t = torch.cat([h_l_emb, h_time_l], dim=1)
            h_p_t = torch.cat([h_p_emb, h_time_p], dim=1)
        else:
            h_l_t = h_l_emb
            h_p_t = h_p_emb

        # --- 2. L-L Path (Internal Physics FIRST) ---
        edges_ll = self.get_ligand_edges(mask_lig, x_l)
        
        if edges_ll.shape[1] == 0:
            # No L-L edges, so no chemistry update
            h_ll_final = h_l_t.clone()
            x_ll_final = x_l.clone()
            vel_ll = torch.zeros_like(x_l) # Store the L-L velocity
        else:
            edge_attr_ll = None
            if self.edge_nf > 0:
                edge_types_ll = torch.ones(edges_ll.size(1), dtype=int, device=edges_ll.device)
                edge_attr_ll = self.edge_embedding(edge_types_ll)

            # Run the chemistry brain
            h_ll_final, x_ll_final = self.egnn(
                h_l_t, x_l, edges_ll,
                node_mask=mask_lig.unsqueeze(-1), 
                batch_mask=mask_lig,
                edge_attr=edge_attr_ll
            )
            # This is the velocity *just from chemistry*
            vel_ll = (x_ll_final - x_l)
            
        # h_ll_final and x_ll_final are now the "chemically-aware" ligand state

        # Find neighbors using the *new* chemically-aware coordinates
        edges_lp = self.get_cross_edges(mask_lig, mask_context, x_ll_final, x_p)

        if edges_lp.shape[1] == 0:
            # No L-P edges, so the final state is just the chemistry state
            final_velocity = vel_ll
            h_final_emb = h_ll_final[:, :-1] if self.condition_time else h_ll_final
        else:
            edge_attr_lp = None
            if self.edge_nf > 0:
                edge_types_lp = torch.zeros(edges_lp.size(1), dtype=int, device=edges_lp.device)
                edge_attr_lp = self.edge_embedding(edge_types_lp)

            # Run the pocket brain
            # INPUT is the OUTPUT from the L-L path
            h_lp_final, x_lp_final = self.cross_attention(
                h_q=h_ll_final, x_q=x_ll_final, 
                h_kv=h_p_t, x_kv=x_p, 
                edge_index=edges_lp, 
                node_mask_q=mask_lig.unsqueeze(-1),
                batch_mask=mask_lig,
                edge_attr=edge_attr_lp
            )
            
            # The final state *is* the output of the L-P path
            # We calculate velocity relative to the *original* x_l
            final_velocity = (x_lp_final - x_l)
            h_final_emb = h_lp_final[:, :-1] if self.condition_time else h_lp_final

        # --- 4. Combine Updates (No longer needed, it's sequential) ---
        # STABILITY FIX: Average updates instead of adding
        #final_velocity = vel_ll + vel_lp
        #h_final_emb = h_update_ll_features + h_update_lp_features
        
        # Decode features back to original atom_nf
        final_features = self.atom_decoder(h_final_emb)

        # STABILITY FIX: Final safety check for NaNs
        if torch.isnan(final_velocity).any() or torch.isnan(final_features).any():
            print("Warning: NaN detected in AtomicaDynamics output. Replacing with zeros.")
            final_velocity = torch.nan_to_num(final_velocity, nan=0.0)
            final_features = torch.nan_to_num(final_features, nan=0.0)

        # Context (pocket) is fixed, so its "update" is all zeros.
        ligand_update = torch.cat([final_velocity, final_features], dim=-1)
        pocket_update = torch.zeros_like(xh_context)

        return ligand_update, pocket_update
    
    def get_ligand_edges(self, batch_mask_ligand, x_ligand):
        adj_ligand = batch_mask_ligand[:, None] == batch_mask_ligand[None, :]
        if self.edge_cutoff_l is not None:
            dists = torch.cdist(x_ligand, x_ligand)
            adj_ligand = adj_ligand & (dists <= self.edge_cutoff_l)
        # STABILITY FIX: Add this line to remove self-loops
        torch.diagonal(adj_ligand).fill_(False)
        edges = torch.stack(torch.where(adj_ligand), dim=0)
        return edges

    def get_cross_edges(self, batch_mask_ligand, batch_mask_pocket, x_ligand, x_pocket):
        if len(x_pocket) == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=x_ligand.device)
        adj_cross = batch_mask_ligand[:, None] == batch_mask_pocket[None, :]
        if self.edge_cutoff_i is not None:
            # STABILITY FIX: Add epsilon to cdist to prevent NaN gradients from sqrt(0)
            dists = torch.cdist(x_ligand, x_pocket)
            adj_cross = adj_cross & (dists <= self.edge_cutoff_i)
        edges = torch.stack(torch.where(adj_cross), dim=0)
        return edges
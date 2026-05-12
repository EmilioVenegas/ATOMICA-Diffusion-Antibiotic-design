import torch
import torch.nn as nn
import torch.nn.functional as F
from equivariant_diffusion.egnn_new import EGNN, GNN
from equivariant_diffusion.en_diffusion import EnVariationalDiffusion
remove_mean_batch = EnVariationalDiffusion.remove_mean_batch
import numpy as np


class SE3EquivariantCrossAttention(nn.Module):
    T_EMBED_DIM = 16  # timestep embedding size (matches saved checkpoints)

    def __init__(self, ligand_nf, pocket_nf, hidden_nf=64, act_fn=nn.SiLU()):
        super().__init__()
        # Timestep embedding: scalar t → T_EMBED_DIM
        self.t_embed = nn.Sequential(
            nn.Linear(1, self.T_EMBED_DIM),
            nn.SiLU(),
            nn.Linear(self.T_EMBED_DIM, self.T_EMBED_DIM),
        )
        query_dim = ligand_nf + self.T_EMBED_DIM
        self.q_proj = nn.Linear(query_dim, hidden_nf)
        self.k_proj = nn.Linear(pocket_nf, hidden_nf)
        self.v_proj = nn.Linear(pocket_nf, hidden_nf)
        self.out_proj = nn.Linear(hidden_nf, ligand_nf)
        self.scale = hidden_nf ** -0.5

        # Timestep-gated output scale: t → gate ∈ (0,1)
        # Learns how much ATOMICA guidance to apply at each noise level.
        self.gate = nn.Sequential(
            nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 1), nn.Sigmoid()
        )
        # Init gate to ~0.5 so adapter starts at half-scale
        nn.init.zeros_(self.gate[2].bias)

        nn.init.xavier_uniform_(self.q_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=0.1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, h_l, h_p, mask_l, mask_p, t=None):
        # 1. Check Embeddings for NaNs
        if torch.isnan(h_l).any(): h_l = torch.nan_to_num(h_l)
        if torch.isnan(h_p).any(): h_p = torch.nan_to_num(h_p)

        # 2. Projections — concatenate timestep embedding to query
        t_per_atom = None
        if t is not None:
            if t.numel() == 1:
                t_per_atom = t.view(1, 1).expand(h_l.shape[0], 1).float()
            else:
                # t is per-molecule; broadcast to per-atom via mask_l
                t_per_atom = t[mask_l].view(-1, 1).float()
            t_emb = self.t_embed(t_per_atom)
            h_l_query = torch.cat([h_l, t_emb], dim=-1)
        else:
            h_l_query = torch.cat([h_l, torch.zeros(h_l.shape[0], self.T_EMBED_DIM,
                                                     device=h_l.device, dtype=h_l.dtype)], dim=-1)
        q = self.q_proj(h_l_query)
        k = self.k_proj(h_p)
        v = self.v_proj(h_p)

        # 3. Attention Scores
        batch_mask = mask_l[:, None] == mask_p[None, :]
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        scores = torch.clamp(scores, min=-100, max=100)
        scores = scores.masked_fill(~batch_mask, -1e4)
        attn = F.softmax(scores, dim=1)

        # Clean up Softmax NaNs
        if torch.isnan(attn).any():
            attn = torch.nan_to_num(attn, nan=0.0)
        attn = attn * batch_mask.float()

        # 4. Feature update gated by timestep
        h_update_internal = torch.matmul(attn, v)
        h_update = self.out_proj(h_update_internal)

        if t_per_atom is not None:
            gate = self.gate(t_per_atom)          # [n_atoms, 1], values in (0,1)
            h_update = h_update * gate

        if torch.isnan(h_update).any():
            h_update = torch.nan_to_num(h_update)

        return h_update

class EGNNDynamics(nn.Module):
    def __init__(self, atom_nf, residue_nf,
                 n_dims, joint_nf=16, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics',
                 norm_constant=0, inv_sublayers=2, sin_embedding=False,
                 normalization_factor=100, aggregation_method='sum',
                 update_pocket_coords=True, edge_cutoff_ligand=None,
                 edge_cutoff_pocket=None, edge_cutoff_interaction=None,
                 reflection_equivariant=True, edge_embedding_dim=None,
                 atomica_nf=None):
        super().__init__()
        self.mode = mode
        self.edge_cutoff_l = edge_cutoff_ligand
        self.edge_cutoff_p = edge_cutoff_pocket
        self.edge_cutoff_i = edge_cutoff_interaction
        self.edge_nf = edge_embedding_dim
        self.atomica_nf = atomica_nf

        if atomica_nf:  # Only create cross_attn if atomica_nf > 0
            # Add LayerNorm for stabilizing attention (with epsilon for numerical stability)
            self.atomica_norm = nn.LayerNorm(atomica_nf, eps=1e-6)

            self.cross_attn = SE3EquivariantCrossAttention(
                ligand_nf=joint_nf,
                pocket_nf=atomica_nf,
                hidden_nf=hidden_nf,
                act_fn=act_fn
            )
            
            self.adapter_scale = nn.Parameter(torch.tensor([0.1]))
        else:
            self.atomica_norm = None
            self.cross_attn = None

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

    def forward(self, xh_atoms, xh_residues, t, mask_atoms, mask_residues, h_atomica=None):

        x_atoms = xh_atoms[:, :self.n_dims].clone()
        h_atoms = xh_atoms[:, self.n_dims:].clone()

        x_residues = xh_residues[:, :self.n_dims].clone()
        h_residues = xh_residues[:, self.n_dims:].clone()

        # embed atom features and residue features in a shared space
        h_atoms = self.atom_encoder(h_atoms)
        h_residues = self.residue_encoder(h_residues)

        # Semantically-Guided Adapter
        vel_adapter = 0.0
        if self.cross_attn is not None and h_atomica is not None:
            # Check for NaN/Inf in input embeddings
            if torch.isnan(h_atomica).any() or torch.isinf(h_atomica).any():
                print(f"WARNING: NaN/Inf in h_atomica before normalization")
                h_atomica = torch.nan_to_num(h_atomica, nan=0.0, posinf=1e4, neginf=-1e4)

            # Normalize ATOMICA embeddings for attention stability
            # Safe LayerNorm: Check for zero-variance (padding) vectors
            if self.cross_attn is not None and h_atomica is not None:
                # Handle NaNs in input
                h_atomica = torch.nan_to_num(h_atomica)
                
                # Apply Norm
                h_atomica_norm = self.atomica_norm(h_atomica)
                
                # FIX: If LayerNorm produced NaNs (due to 0-variance inputs), replace them
                if torch.isnan(h_atomica_norm).any():
                    h_atomica_norm = torch.nan_to_num(h_atomica_norm)
            
            # Check after normalization
            if torch.isnan(h_atomica_norm).any() or torch.isinf(h_atomica_norm).any():
                print(f"WARNING: NaN/Inf in h_atomica after normalization")
                h_atomica_norm = torch.nan_to_num(h_atomica_norm, nan=0.0, posinf=1e4, neginf=-1e4)
            
            h_update = self.cross_attn(
                h_l=h_atoms,
                h_p=h_atomica_norm,
                mask_l=mask_atoms,
                mask_p=mask_residues,
                t=t,
            )
            if torch.isnan(h_update).any():
                h_update = torch.nan_to_num(h_update, nan=0.0)
            h_atoms = h_atoms + self.adapter_scale * h_update

        # combine the two node types
        x = torch.cat((x_atoms, x_residues), dim=0)
        h = torch.cat((h_atoms, h_residues), dim=0)
        mask = torch.cat([mask_atoms, mask_residues])

        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t[mask]
            h = torch.cat([h, h_time], dim=1)

        # get edges of a complete graph
        edges = self.get_edges(mask_atoms, mask_residues, x_atoms, x_residues)
        assert torch.all(mask[edges[0]] == mask[edges[1]])

        # Get edge types
        if self.edge_nf > 0:
            # 0: ligand-pocket, 1: ligand-ligand, 2: pocket-pocket
            edge_types = torch.zeros(edges.size(1), dtype=int, device=edges.device)
            edge_types[(edges[0] < len(mask_atoms)) & (edges[1] < len(mask_atoms))] = 1
            edge_types[(edges[0] >= len(mask_atoms)) & (edges[1] >= len(mask_atoms))] = 2

            # Learnable embedding
            edge_types = self.edge_embedding(edge_types)
        else:
            edge_types = None

        if self.mode == 'egnn_dynamics':
            update_coords_mask = None if self.update_pocket_coords \
                else torch.cat((torch.ones_like(mask_atoms),
                                torch.zeros_like(mask_residues))).unsqueeze(1)
            h_final, x_final = self.egnn(h, x, edges,
                                         update_coords_mask=update_coords_mask,
                                         batch_mask=mask, edge_attr=edge_types)
            vel = (x_final - x)

        elif self.mode == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=None, edge_attr=edge_types)
            vel = output[:, :3]
            h_final = output[:, 3:]

        else:
            raise Exception("Wrong mode %s" % self.mode)

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]

        # decode atom and residue features
        h_final_atoms = self.atom_decoder(h_final[:len(mask_atoms)])
        h_final_residues = self.residue_decoder(h_final[len(mask_atoms):])

        if torch.any(torch.isnan(vel)):
            if self.training:
                raise RuntimeError("NaN detected in EGNN output during training!")
            else:
                vel[torch.isnan(vel)] = 0.0
                print("WARNING: NaN detected in EGNN output during validation - replacing with zeros")

        if self.update_pocket_coords:
            # in case of unconditional joint distribution, include this as in
            # the original code
            vel = remove_mean_batch(vel, mask)

        return torch.cat([vel[:len(mask_atoms)], h_final_atoms], dim=-1), \
               torch.cat([vel[len(mask_atoms):], h_final_residues], dim=-1)

    def get_edges(self, batch_mask_ligand, batch_mask_pocket, x_ligand, x_pocket):
        # Optimized: Block-Diagonal Graph Construction
        edges_list = []
        
        # Get unique batch indices
        n_batches = int(batch_mask_ligand.max()) + 1
        
        # Pre-calculate the offset for pocket atoms in the global concatenated tensor
        # Global structure is: [Ligand_Atoms ... | Pocket_Atoms ...]
        pocket_offset = len(batch_mask_ligand)
        
        for i in range(n_batches):
            # 1. Find indices for this batch item
            ligand_indices = torch.nonzero(batch_mask_ligand == i).squeeze(-1)
            pocket_indices = torch.nonzero(batch_mask_pocket == i).squeeze(-1)
            
            # Skip empty batches (if any)
            if len(ligand_indices) == 0 and len(pocket_indices) == 0:
                continue

            # 2. Extract Coordinates for this batch
            x_l_batch = x_ligand[ligand_indices]
            x_p_batch = x_pocket[pocket_indices]
            
            # 3. Compute Adjacency (Local)
            # Ligand-Ligand
            if self.edge_cutoff_l is not None:
                d_ll = torch.cdist(x_l_batch, x_l_batch)
                adj_ll = d_ll <= self.edge_cutoff_l
            else:
                adj_ll = torch.ones(len(x_l_batch), len(x_l_batch), device=x_l_batch.device, dtype=torch.bool)
                
            # Pocket-Pocket
            if self.edge_cutoff_p is not None:
                d_pp = torch.cdist(x_p_batch, x_p_batch)
                adj_pp = d_pp <= self.edge_cutoff_p
            else:
                adj_pp = torch.ones(len(x_p_batch), len(x_p_batch), device=x_p_batch.device, dtype=torch.bool)
                
            # Cross (Ligand-Pocket)
            if self.edge_cutoff_i is not None:
                d_lp = torch.cdist(x_l_batch, x_p_batch)
                adj_lp = d_lp <= self.edge_cutoff_i
            else:
                adj_lp = torch.ones(len(x_l_batch), len(x_p_batch), device=x_l_batch.device, dtype=torch.bool)

            # 4. Build Block Matrix
            top = torch.cat([adj_ll, adj_lp], dim=1)
            bottom = torch.cat([adj_lp.t(), adj_pp], dim=1)
            adj_batch = torch.cat([top, bottom], dim=0)
            
            # 5. Convert to Edge Indices (Local to this batch's subset)
            src_local, dst_local = torch.where(adj_batch)
            
            # 6. Map Local Indices -> Global Indices
            # CRITICAL FIX: Add offset to pocket indices so they point to the correct
            # section of the concatenated [Ligand, Pocket] tensors.
            
            global_indices = torch.cat([ligand_indices, pocket_indices + pocket_offset])
            
            src_global = global_indices[src_local]
            dst_global = global_indices[dst_local]
            
            edges_batch = torch.stack([src_global, dst_global], dim=0)
            edges_list.append(edges_batch)

        if len(edges_list) > 0:
            edges = torch.cat(edges_list, dim=1)
        else:
            edges = torch.empty((2, 0), device=x_ligand.device, dtype=torch.long)

        return edges

    def freeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = False
        
        if self.cross_attn is not None:
            for param in self.cross_attn.parameters():
                param.requires_grad = True
        
        for param in self.residue_encoder.parameters():
            param.requires_grad = True

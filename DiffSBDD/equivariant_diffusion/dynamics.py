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

    def forward(self, xh_atoms, xh_residues, t, mask_atoms, mask_residues):

        x_atoms = xh_atoms[:, :self.n_dims].clone()
        h_atoms = xh_atoms[:, self.n_dims:].clone()

        x_residues = xh_residues[:, :self.n_dims].clone()
        h_residues = xh_residues[:, self.n_dims:].clone()

        # embed atom features and residue features in a shared space
        h_atoms = self.atom_encoder(h_atoms)
        h_residues = self.residue_encoder(h_residues)

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
                vel[torch.isnan(vel)] = 0.0
            else:
                raise ValueError("NaN detected in EGNN output")

        if self.update_pocket_coords:
            # in case of unconditional joint distribution, include this as in
            # the original code
            vel = remove_mean_batch(vel, mask)

        return torch.cat([vel[:len(mask_atoms)], h_final_atoms], dim=-1), \
               torch.cat([vel[len(mask_atoms):], h_final_residues], dim=-1)

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

        # Ligand (atom) feature encoder
        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_nf, 2 * atom_nf),
            act_fn,
            nn.Linear(2 * atom_nf, hidden_nf)
        )

        # Ligand (atom) feature decoder
        self.atom_decoder = nn.Sequential(
            nn.Linear(hidden_nf, 2 * atom_nf),
            act_fn,
            nn.Linear(2 * atom_nf, atom_nf)
        )

        # Pocket (context) embedding encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(context_nf, 2 * context_nf),
            act_fn,
            nn.Linear(2 * context_nf, hidden_nf)
        )

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
            n_layers=n_layers, attention=attention, tanh=tanh,
            norm_constant=norm_constant,
            inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
            reflection_equiv=reflection_equivariant
        )
        
        # 2. Ligand-Pocket (L-P) Module
        self.cross_attention = SE3CrossAttention(
            in_node_nf_q=dynamics_node_nf, in_node_nf_kv=hidden_nf, # KV (pocket) is not time-conditioned
            in_edge_nf=self.edge_nf,
            hidden_nf=hidden_nf, device=device, act_fn=act_fn,
            n_layers=n_layers, attention=attention, tanh=tanh,
            norm_constant=norm_constant,
            inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
            reflection_equiv=reflection_equivariant
        )
        # --- End Core Components ---

    def forward(self, xh_lig, xh_context, t, mask_lig, mask_context):

        # --- 1. Prepare Inputs ---
        x_l = xh_lig[:, :self.n_dims].clone()
        h_l = xh_lig[:, self.n_dims:].clone()

        x_p = xh_context[:, :self.n_dims].clone()
        h_p_atomica = xh_context[:, self.n_dims:].clone() # These are the ATOMICA embeddings

        # Embed features
        h_l_emb = self.atom_encoder(h_l)
        h_p_emb = self.context_encoder(h_p_atomica)

        # Condition ligand features on time
        if self.condition_time:
            if np.prod(t.size()) == 1:
                h_time = torch.empty_like(h_l_emb[:, 0:1]).fill_(t.item())
            else:
                h_time = t[mask_lig]
            h_l_t = torch.cat([h_l_emb, h_time], dim=1)
        else:
            h_l_t = h_l_emb
            
        # h_p_emb is not time-conditioned, it's fixed context

        # --- 2. L-L Path (Internal Physics) ---
        edges_ll = self.get_ligand_edges(mask_lig, x_l)
        
        edge_attr_ll = None
        if self.edge_nf > 0:
            edge_types_ll = torch.ones(edges_ll.size(1), dtype=int, device=edges_ll.device) # Type 1 for L-L
            edge_attr_ll = self.edge_embedding(edge_types_ll)

        h_ll_final, x_ll_final = self.egnn(
            h_l_t, x_l, edges_ll,
            node_mask=mask_lig.unsqueeze(-1), 
            edge_attr=edge_attr_ll
        )
        
        vel_ll = (x_ll_final - x_l)
        h_update_ll = (h_ll_final - h_l_t) if self.condition_time else (h_ll_final - h_l_emb)


        # --- 3. L-P Path (Cross-Attention Interaction) ---
        edges_lp = self.get_cross_edges(mask_lig, mask_context, x_l, x_p)

        edge_attr_lp = None
        if self.edge_nf > 0:
            edge_types_lp = torch.zeros(edges_lp.size(1), dtype=int, device=edges_lp.device) # Type 0 for L-P
            edge_attr_lp = self.edge_embedding(edge_types_lp)

        h_lp_final, x_lp_final = self.cross_attention(
            h_q=h_l_t, x_q=x_l, 
            h_kv=h_p_emb, x_kv=x_p, 
            edge_index=edges_lp,
            node_mask_q=mask_lig.unsqueeze(-1), 
            edge_attr=edge_attr_lp
        )

        vel_lp = (x_lp_final - x_l)
        h_update_lp = (h_lp_final - h_l_t) if self.condition_time else (h_lp_final - h_l_emb)

        # --- 4. Combine Updates ---
        final_velocity = vel_ll + vel_lp
        
        # Combine feature updates
        h_final_emb = h_l_emb + h_update_ll + h_update_lp
        
        # Decode features back to original atom_nf
        final_features = self.atom_decoder(h_final_emb)

        if torch.any(torch.isnan(final_velocity)):
            if self.training:
                final_velocity[torch.isnan(final_velocity)] = 0.0
            else:
                raise ValueError("NaN detected in AtomicaDynamics output velocity")
        if torch.any(torch.isnan(final_features)):
             if self.training:
                final_features[torch.isnan(final_features)] = 0.0
             else:
                raise ValueError("NaN detected in AtomicaDynamics output features")

        # Context (pocket) is fixed, so its "update" is all zeros.
        # This matches the EGNNDynamics return signature.
        ligand_update = torch.cat([final_velocity, final_features], dim=-1)
        pocket_update = torch.zeros_like(xh_context)

        return ligand_update, pocket_update


    def get_ligand_edges(self, batch_mask_ligand, x_ligand):
        adj_ligand = batch_mask_ligand[:, None] == batch_mask_ligand[None, :]
        if self.edge_cutoff_l is not None:
            adj_ligand = adj_ligand & (torch.cdist(x_ligand, x_ligand) <= self.edge_cutoff_l)
        adj_ligand = adj_ligand ^ torch.diag(torch.diag(adj_ligand)) # remove self-loops
        edges = torch.stack(torch.where(adj_ligand), dim=0)
        return edges

    def get_cross_edges(self, batch_mask_ligand, batch_mask_pocket, x_ligand, x_pocket):
        adj_cross = batch_mask_ligand[:, None] == batch_mask_pocket[None, :]
        if self.edge_cutoff_i is not None:
            adj_cross = adj_cross & (torch.cdist(x_ligand, x_pocket) <= self.edge_cutoff_i)
        edges = torch.stack(torch.where(adj_cross), dim=0)
        return edges
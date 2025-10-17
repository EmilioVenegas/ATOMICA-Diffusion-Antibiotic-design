import torch
import torch.nn as nn
import torch.nn.functional as F
from equivariant_diffusion.egnn_new import EGNN, GNN
from equivariant_diffusion.en_diffusion import EnVariationalDiffusion
# Assuming cross_attention.py is in the path
from cross_attention import SE3EquivariantCrossAttention 
remove_mean_batch = EnVariationalDiffusion.remove_mean_batch
import numpy as np


class EGNNDynamics(nn.Module):
    def __init__(self, atom_nf, residue_nf, # residue_nf will now be atomica_dim
                 n_dims, joint_nf=16, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics',
                 norm_constant=0, inv_sublayers=2, sin_embedding=False,
                 normalization_factor=100, aggregation_method='sum',
                 update_pocket_coords=True, edge_cutoff_ligand=None,
                 edge_cutoff_pocket=None, edge_cutoff_interaction=None,
                 reflection_equivariant=True, edge_embedding_dim=None,
                 ### MODIFICATION START ###
                 use_cross_attention=True,
                 atomica_dim=128, # Example dimension for ATOMICA embeddings
                 cross_attention_heads=4,
                 cross_attention_cutoff=10.0
                 ### MOD-END ###
                ):
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

        ### MOD start ###
        # This encoder will now process the ATOMICA embeddings
        # Check input dimensions
        self.residue_encoder = nn.Sequential(
            nn.Linear(atomica_dim, 2 * atomica_dim),
            act_fn,
            nn.Linear(2 * atomica_dim, joint_nf)
        )
        ### MOD-END ###

        self.residue_decoder = nn.Sequential(
            nn.Linear(joint_nf, 2 * residue_nf),
            act_fn,
            nn.Linear(2 * residue_nf, residue_nf)
        )
        
        ### mod START ###
        # Instantiate the Cross-Attention Module
        self.use_cross_attention = use_cross_attention
        if self.use_cross_attention:
            self.cross_attention = SE3EquivariantCrossAttention(
                hidden_nf=joint_nf, # It operates on the encoded features
                atomica_dim=atomica_dim,
                num_heads=cross_attention_heads,
                edge_cutoff=cross_attention_cutoff
            )
        ### MOD-END ###

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

    def forward(self, xh_atoms, xh_residues, t, mask_atoms, mask_residues, 
                ### mod START ###
                h_atomica
                ### MOD-END ###
               ):

        x_atoms = xh_atoms[:, :self.n_dims].clone()
        h_atoms = xh_atoms[:, self.n_dims:].clone()

        x_residues = xh_residues[:, :self.n_dims].clone()
        # h_residues is no longer used, h_atomica is used instead
        # h_residues = xh_residues[:, self.n_dims:].clone()

        # embed atom features and residue features in a shared space
        h_atoms_encoded = self.atom_encoder(h_atoms)
        
        ### MODIFICATION START ###
        # Note: h_residues from input is ignored. We now use h_atomica.
        # h_atomica is the raw ATOMICA embedding for the pocket.
        
        # We need an encoded version for the main EGNN, but the cross-attention
        # module takes the raw ATOMICA embedding.
        h_residues_encoded = self.residue_encoder(h_atomica)
        
        # Initial displacement from cross-attention
        vel_cross_attention = torch.zeros_like(x_atoms)

        if self.use_cross_attention:
            # The cross-attention module updates the ligand based on the pocket
            h_atoms_attended, x_atoms_attended = self.cross_attention(
                h_lig=h_atoms_encoded,
                x_lig=x_atoms,
                h_atomica=h_atomica,
                x_pocket=x_residues
            )
            # The output of the cross-attention becomes the new input for the ligand
            h_atoms_encoded = h_atoms_attended
            
            #  capture the coordinate update from this step
            vel_cross_attention = x_atoms_attended - x_atoms
            
            # The main EGNN will start from the original ligand coordinates
            # and predict a further update.
            x_atoms_for_egnn = x_atoms
        else:
            x_atoms_for_egnn = x_atoms
        ### MOD-END ###


        # combine the two node types for the main EGNN
        x = torch.cat((x_atoms_for_egnn, x_residues), dim=0)
        h = torch.cat((h_atoms_encoded, h_residues_encoded), dim=0)
        mask = torch.cat([mask_atoms, mask_residues])

        if self.condition_time:
            if np.prod(t.size()) == 1:
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                h_time = t[mask]
            h = torch.cat([h, h_time], dim=1)

        # get edges of a complete graph
        edges = self.get_edges(mask_atoms, mask_residues, x_atoms, x_residues)
        assert torch.all(mask[edges[0]] == mask[edges[1]])

        if self.edge_nf > 0:
            edge_types = torch.zeros(edges.size(1), dtype=int, device=edges.device)
            edge_types[(edges[0] < len(mask_atoms)) & (edges[1] < len(mask_atoms))] = 1
            edge_types[(edges[0] >= len(mask_atoms)) & (edges[1] >= len(mask_atoms))] = 2
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
            vel_egnn = (x_final - x)

        elif self.mode == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=None, edge_attr=edge_types)
            vel_egnn = output[:, :3]
            h_final = output[:, 3:]
        else:
            raise Exception("Wrong mode %s" % self.mode)

        ### mod START ###
        # The final velocity is the sum of the update from cross-attention 
        # and the update from the main EGNN message passing.
        vel = vel_egnn
        vel[:len(mask_atoms)] += vel_cross_attention
        ### MOD-END ###

        if self.condition_time:
            h_final = h_final[:, :-1]

        h_final_atoms = self.atom_decoder(h_final[:len(mask_atoms)])
        # The residue decoder output dimension must match the original residue_nf
        
        h_final_residues = self.residue_decoder(h_final[len(mask_atoms):])

        if torch.any(torch.isnan(vel)):
            if self.training:
                vel[torch.isnan(vel)] = 0.0
            else:
                raise ValueError("NaN detected in EGNN output")

        if self.update_pocket_coords:
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
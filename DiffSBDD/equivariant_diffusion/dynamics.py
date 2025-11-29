import torch
import torch.nn as nn
import torch.nn.functional as F
from equivariant_diffusion.egnn_new import EGNN, GNN
from equivariant_diffusion.en_diffusion import EnVariationalDiffusion
remove_mean_batch = EnVariationalDiffusion.remove_mean_batch
import numpy as np


class SE3EquivariantCrossAttention(nn.Module):
    def __init__(self, ligand_nf, pocket_nf, hidden_nf=64, act_fn=nn.SiLU()):
        super().__init__()
        self.q_proj = nn.Linear(ligand_nf, hidden_nf)
        self.k_proj = nn.Linear(pocket_nf, hidden_nf)
        self.v_proj = nn.Linear(pocket_nf, hidden_nf)
        self.out_proj = nn.Linear(hidden_nf, ligand_nf)
        self.act = act_fn
        self.scale = hidden_nf ** -0.5

    def forward(self, h_l, x_l, h_p, x_p, mask_l, mask_p):
        # h_l: (N_l, D_l), x_l: (N_l, 3)
        # h_p: (N_p, D_p), x_p: (N_p, 3)

        # 1. Compute Queries, Keys, Values
        q = self.q_proj(h_l)  # (N_l, H)
        k = self.k_proj(h_p)  # (N_p, H)
        v = self.v_proj(h_p)  # (N_p, H)

        # 2. Compute Attention Scores
        # Construct mask for valid pairs (same batch index)
        # mask_l: (N_l,), mask_p: (N_p,)
        # batch_mask: (N_l, N_p) where True if same batch
        batch_mask = mask_l[:, None] == mask_p[None, :]

        # Compute dense scores (N_l, N_p)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (N_l, N_p)

        # Mask out invalid pairs (different batches)
        scores = scores.masked_fill(~batch_mask, -1e4)

        # Softmax over pocket dimension (dim=1)
        attn = F.softmax(scores, dim=1)  # (N_l, N_p)

        # Mask again to ensure zero attention for invalid pairs
        attn = attn * batch_mask.float()

        # 3. Semantic Update
        h_update = torch.matmul(attn, v)  # (N_l, H)
        h_update = self.out_proj(h_update)  # (N_l, D_l)

        # 4. Coordinate Update (Equivariant)
        # vel_i = sum_j attn_ij * (x_l_i - x_p_j)
        # This is equivalent to: x_l - weighted_center_of_pocket
        center_p = torch.matmul(attn, x_p)  # (N_l, 3)
        vel = x_l - center_p

        return h_update, vel

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

        if atomica_nf is not None:
            self.cross_attn = SE3EquivariantCrossAttention(
                ligand_nf=joint_nf,
                pocket_nf=atomica_nf,
                hidden_nf=hidden_nf,
                act_fn=act_fn
            )
        else:
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
            # h_atoms is the ligand features (before shared encoder? No, user said "ligand features" and "pocket ATOMICA embeddings")
            # The user said: "Pass ligand features and h_atomica into the Cross-Attention adapter before the main EGNN."
            # h_atoms here is the raw input features (one-hot etc).
            # The adapter should probably operate on the raw features or the encoded ones?
            # "It should take ligand features and pocket ATOMICA embeddings as input."
            # "compute a semantic feature update for the ligand."
            # If we update h_atoms here, it will be passed to atom_encoder.
            # But h_atomica is "rich embeddings".
            # Let's assume we pass the raw h_atoms (one-hot) to the adapter?
            # Or maybe we should encode h_atoms first?
            # The atom_encoder projects to 'joint_nf'.
            # If we update h_atoms, we change the input to atom_encoder.
            # But h_update from adapter might not be same dim as h_atoms (one-hot).
            # Let's look at dimensions.
            # h_atoms: atom_nf. h_atomica: atomica_nf.
            # Adapter output h_update: atom_nf (to add to h_atoms).
            # This seems correct.

            h_update, vel_adapter = self.cross_attn(
                h_l=h_atoms,
                x_l=x_atoms,
                h_p=h_atomica,
                x_p=x_residues,
                mask_l=mask_atoms,
                mask_p=mask_residues
            )
            h_atoms = h_atoms + h_update

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
            if self.cross_attn is not None:
                # Add adapter velocity to ligand velocity
                # vel is (N_total, 3). Ligand is first len(mask_atoms).
                # vel_adapter is (N_l, 3).
                vel[:len(mask_atoms)] = vel[:len(mask_atoms)] + vel_adapter

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
            vel[torch.isnan(vel)] = 0.0
            if not self.training:
                print("WARNING: NaN detected in EGNN output during validation - replacing with zeros")

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

    def freeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = False
        
        if self.cross_attn is not None:
            for param in self.cross_attn.parameters():
                param.requires_grad = True
        
        for param in self.residue_encoder.parameters():
            param.requires_grad = True

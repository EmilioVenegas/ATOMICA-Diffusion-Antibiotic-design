import math
from argparse import Namespace
from typing import Optional
from time import time
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau # Or another scheduler
import pytorch_lightning as pl
import wandb
from torch_scatter import scatter_add, scatter_mean
from Bio.PDB import PDBParser


from constants import dataset_params, FLOAT_TYPE, INT_TYPE
from equivariant_diffusion.dynamics import EGNNDynamics
# --- NEW: Import AtomicaDynamics ---
from equivariant_diffusion.dynamics import AtomicaDynamics
from equivariant_diffusion.en_diffusion import EnVariationalDiffusion
from equivariant_diffusion.conditional_model import ConditionalDDPM, \
    SimpleConditionalDDPM
from dataset import ProcessedLigandPocketDataset
import utils
from analysis.visualization import save_xyz_file, visualize, visualize_chain
from analysis.metrics import BasicMolecularMetrics, CategoricalDistribution, \
    MoleculeProperties
from analysis.molecule_builder import build_molecule, process_molecule
from analysis.docking import smina_score
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, SequentialLR
from analysis.visualization import save_xyz_file, plot_molecule_and_pocket


# --- NEW: Imports for Phase 4 (Inference) ---
try:
    from ATOMICA.models.prediction_model import PredictionModel
    from ATOMICA.models.pretrain_model import DenoisePretrainModel
    from ATOMICA.models.prot_interface_model import ProteinInterfaceModel
    from ATOMICA.data.pdb_utils import VOCAB
    # --- NEW: Re-use data processing logic from Phase 1 ---
    from utils import format_atomica_batch, load_atomica_model
    ATOMICA_IMPORTS_OK = True
except ImportError as e:
    print(f"Warning: Could not import ATOMICA modules: {e}. "
          "Training (Phase 3) will work, but "
          "Inference (Phase 4) via generate_ligands() will fail.")
    ATOMICA_IMPORTS_OK = False

PROTEIN_LETTERS_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEP": "S", "TPO": "T", "PTR": "Y", "CSO": "C", 
    "SEC": "U", "PYL": "O", "UNK": "X",
}

class LigandPocketDDPM(pl.LightningModule):
    def __init__(
            self,
            outdir,
            dataset,
            datadir,
            batch_size,
            lr,
            egnn_params: Namespace,
            diffusion_params,
            num_workers,
            augment_noise,
            augment_rotation,
            clip_grad,
            eval_epochs,
            eval_params,
            visualize_sample_epoch,
            visualize_chain_epoch,
            auxiliary_loss,
            loss_params,
            mode,
            node_histogram,
            pocket_representation='CA',
            virtual_nodes=False,
            # --- NEW: Args for ATOMICA model (Phase 4) ---
            atomica_model_path=None,
            atomica_model_config=None,
            atomica_model_weights=None,
            warmup_steps=500,
            bond_params: Optional[Namespace] = None,
            coord_loss_weight: float = 1.0,
            **kwargs # Add kwargs to catch any other stray arguments
            
    ):
        super(LigandPocketDDPM, self).__init__()
        self.save_hyperparameters()
        

        ddpm_models = {'joint': EnVariationalDiffusion,
                       'pocket_conditioning': ConditionalDDPM,
                       'pocket_conditioning_simple': SimpleConditionalDDPM}
        assert mode in ddpm_models
        self.mode = mode
        # --- MODIFIED: Add 'atomica' as a valid representation ---
        assert pocket_representation in {'CA', 'full-atom', 'atomica'}
        self.pocket_representation = pocket_representation

        self.dataset_name = dataset
        self.datadir = datadir
        self.outdir = outdir
        self.batch_size = batch_size
        self.eval_batch_size = eval_params.eval_batch_size \
            if 'eval_batch_size' in eval_params else batch_size
        self.lr = lr
        self.loss_type = diffusion_params.diffusion_loss_type
        self.eval_epochs = eval_epochs
        self.visualize_sample_epoch = visualize_sample_epoch
        self.visualize_chain_epoch = visualize_chain_epoch
        self.eval_params = eval_params
        self.num_workers = num_workers
        self.augment_noise = augment_noise
        self.augment_rotation = augment_rotation
        self.dataset_info = dataset_params[dataset]
        self.T = diffusion_params.diffusion_steps
        self.clip_grad = clip_grad
    
        self.lig_type_encoder = self.dataset_info['atom_encoder']
        self.lig_type_decoder = self.dataset_info['atom_decoder']
        self.warmup_steps = warmup_steps # Store it
        
        # --- MODIFIED: Handle 'atomica' representation ---
        if self.pocket_representation == 'CA':
            self.pocket_type_encoder = self.dataset_info['aa_encoder']
            self.pocket_type_decoder = self.dataset_info['aa_decoder']
        elif self.pocket_representation == 'full-atom':
            self.pocket_type_encoder = self.dataset_info['atom_encoder']
            self.pocket_type_decoder = self.dataset_info['atom_decoder']
        elif self.pocket_representation == 'atomica':
            # Encoders/Decoders are not used, pocket features are embeddings
            self.pocket_type_encoder = None
            self.pocket_type_decoder = None

        smiles_list = None if eval_params.smiles_file is None \
            else np.load(eval_params.smiles_file)
        self.ligand_metrics = BasicMolecularMetrics(self.dataset_info,
                                                    smiles_list)
        self.molecule_properties = MoleculeProperties()
        self.ligand_type_distribution = CategoricalDistribution(
            self.dataset_info['atom_hist'], self.lig_type_encoder)
        
        # --- MODIFIED: Handle 'atomica' representation ---
        if self.pocket_representation == 'CA':
            self.pocket_type_distribution = CategoricalDistribution(
                self.dataset_info['aa_hist'], self.pocket_type_encoder)
        else: # 'full-atom' or 'atomica'
            self.pocket_type_distribution = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.virtual_nodes = virtual_nodes
        self.data_transform = None
        self.max_num_nodes = len(node_histogram) - 1
        if virtual_nodes:
            # symbol = 'virtual'
            symbol = 'Ne'  # visualize as Neon atoms
            self.lig_type_encoder[symbol] = len(self.lig_type_encoder)
            self.virtual_atom = self.lig_type_encoder[symbol]
            self.lig_type_decoder.append(symbol)
            self.data_transform = utils.AppendVirtualNodes(
                self.max_num_nodes, self.lig_type_encoder, symbol)

            # Update dataset_info dictionary. This is necessary for using the
            # visualization functions.
            self.dataset_info['atom_encoder'] = self.lig_type_encoder
            self.dataset_info['atom_decoder'] = self.lig_type_decoder

            # ADD THESE TWO LINES TO FIX THE CRASH
            self.dataset_info['colors_dic'].append('#9400D3')  # Add a color for the virtual atom (e.g., DarkViolet)
            self.dataset_info['radius_dic'].append(0.5)        # Add a radius for the virtual atom

        self.atom_nf = len(self.lig_type_decoder)
        
        # --- MODIFIED: Set aa_nf based on representation ---
        if self.pocket_representation == 'atomica':
            # aa_nf is now the dimension of the ATOMICA embeddings
            try:
                embed_dim = egnn_params.atomica_embed_dim
                one_hot_dim = egnn_params.atomica_one_hot_dim
                self.aa_nf = embed_dim + one_hot_dim # e.g., 32 + 9 = 41
            except AttributeError as e:
                raise AttributeError(f"egnn_params must include 'atomica_embed_dim' "
                                     f"and 'atomica_one_hot_dim'. Error: {e}")
        else:
            self.aa_nf = len(self.pocket_type_decoder)
            
        self.x_dims = 3

        # --- MODIFIED: Instantiate AtomicaDynamics or EGNNDynamics ---
        if self.pocket_representation == 'atomica':
            print("Using AtomicaDynamics (Phase 2 Model)")
            net_dynamics = AtomicaDynamics(
                atom_nf=self.atom_nf,
                context_nf=self.aa_nf, # This is atomica_embed_dim
                n_dims=self.x_dims,
                hidden_nf=egnn_params.hidden_nf,
                device=egnn_params.device if torch.cuda.is_available() else 'cpu',
                act_fn=torch.nn.SiLU(),
                n_layers=egnn_params.n_layers,
                attention=egnn_params.attention,
                tanh=egnn_params.tanh,
                norm_constant=egnn_params.norm_constant,
                inv_sublayers=egnn_params.inv_sublayers,
                sin_embedding=egnn_params.sin_embedding,
                normalization_factor=egnn_params.normalization_factor,
                aggregation_method=egnn_params.aggregation_method,
                edge_cutoff_ligand=egnn_params.__dict__.get('edge_cutoff_ligand'),
                edge_cutoff_interaction=egnn_params.__dict__.get('edge_cutoff_interaction'),
                reflection_equivariant=egnn_params.reflection_equivariant,
                edge_embedding_dim=egnn_params.__dict__.get('edge_embedding_dim')
            
            )
        else:
            print("Using EGNNDynamics (Original Model)")
            net_dynamics = EGNNDynamics(
                atom_nf=self.atom_nf,
                residue_nf=self.aa_nf,
                n_dims=self.x_dims,
                joint_nf=egnn_params.joint_nf,
                device=egnn_params.device if torch.cuda.is_available() else 'cpu',
                hidden_nf=egnn_params.hidden_nf,
                act_fn=torch.nn.SiLU(),
                n_layers=egnn_params.n_layers,
                attention=egnn_params.attention,
                tanh=egnn_params.tanh,
                norm_constant=egnn_params.norm_constant,
                inv_sublayers=egnn_params.inv_sublayers,
                sin_embedding=egnn_params.sin_embedding,
                normalization_factor=egnn_params.normalization_factor,
                aggregation_method=egnn_params.aggregation_method,
                edge_cutoff_ligand=egnn_params.__dict__.get('edge_cutoff_ligand'),
                edge_cutoff_pocket=egnn_params.__dict__.get('edge_cutoff_pocket'),
                edge_cutoff_interaction=egnn_params.__dict__.get('edge_cutoff_interaction'),
                update_pocket_coords=(self.mode == 'joint'),
                reflection_equivariant=egnn_params.reflection_equivariant,
                edge_embedding_dim=egnn_params.__dict__.get('edge_embedding_dim'),
            )
        

        self.ddpm = ddpm_models[self.mode](
                dynamics=net_dynamics,
                atom_nf=self.atom_nf,
                residue_nf=self.aa_nf, 
                n_dims=self.x_dims,
                timesteps=diffusion_params.diffusion_steps,
                noise_schedule=diffusion_params.diffusion_noise_schedule,
                noise_precision=diffusion_params.diffusion_noise_precision,
                loss_type=diffusion_params.diffusion_loss_type,
                norm_values=diffusion_params.normalize_factors,
                size_histogram=node_histogram,
                virtual_node_idx=self.lig_type_encoder[symbol] if virtual_nodes else None
        )

        self.auxiliary_loss = auxiliary_loss
        self.lj_rm = self.dataset_info['lennard_jones_rm']
        if self.auxiliary_loss:
            self.clamp_lj = loss_params.clamp_lj
            self.auxiliary_weight_schedule = WeightSchedule(
                T=diffusion_params.diffusion_steps,
                max_weight=loss_params.max_weight, mode=loss_params.schedule)

        # --- NEW: For Phase 4 Inference ---
        self.atomica_model = None
        if ATOMICA_IMPORTS_OK:
            self.atomica_vocab = VOCAB
        else:
            self.atomica_vocab = None
            if self.pocket_representation == 'atomica':
                raise ImportError("ATOMICA modules failed to import, but "
                                  "pocket_representation is set to 'atomica'.")

    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.ddpm.parameters(), lr=self.lr, amsgrad=True, weight_decay=1e-12)

        # Define ONLY the plateau scheduler
        plateau_scheduler = ReduceLROnPlateau(
            optimizer, 'min', factor=0.8, patience=10, verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": plateau_scheduler,
                "monitor": "loss/val",  # This is what it will monitor
                "interval": "epoch",    # It will step at the end of each epoch
            },
        }


    def setup(self, stage: Optional[str] = None):
        # --- MODIFIED: Load from processed directory, not .npz ---
        if self.pocket_representation == 'atomica':
            train_path = Path(self.datadir, 'train')
            val_path = Path(self.datadir, 'val')
            test_path = Path(self.datadir, 'test')
            if not train_path.is_dir() or not val_path.is_dir():
                print(f"Warning: '{train_path}' or '{val_path}' not found.")
                print("Make sure your 'datadir' points to the *parent* directory "
                      "containing 'train', 'val', etc. subdirectories "
                      "filled with .pt files from Phase 1.")
        else:
            train_path = Path(self.datadir, 'train.npz')
            val_path = Path(self.datadir, 'val.npz')
            test_path = Path(self.datadir, 'test.npz')
            
        if stage == 'fit':
            self.train_dataset = ProcessedLigandPocketDataset(
                train_path, transform=self.data_transform)
            self.val_dataset = ProcessedLigandPocketDataset(
                val_path, transform=self.data_transform)
        elif stage == 'test':
            self.test_dataset = ProcessedLigandPocketDataset(
                test_path, transform=self.data_transform)
        else:
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=self.train_dataset.collate_fn,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.val_dataset.collate_fn,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.test_dataset.collate_fn,
                          pin_memory=True)

    def get_ligand_and_pocket(self, data):
        ligand = {
            'x': data['lig_coords'].to(self.device, FLOAT_TYPE),
            'one_hot': data['lig_one_hot'].to(self.device, FLOAT_TYPE),
            'size': data['num_lig_atoms'].to(self.device, INT_TYPE),
            'mask': data['lig_mask'].to(self.device, INT_TYPE),
        }
        if self.virtual_nodes:
            ligand['num_virtual_atoms'] = data['num_virtual_atoms'].to(
                self.device, INT_TYPE)
        
        if self.pocket_representation == 'atomica':
            # --- Load both embedding and one-hot features ---
            embed_feats = data['pocket_atomica_embeddings'].to(self.device, FLOAT_TYPE)
            one_hot_feats = data['pocket_one_hot'].to(self.device, FLOAT_TYPE)
            
            # --- Concatenate features ---
            pocket_features = torch.cat([embed_feats, one_hot_feats], dim=1)
        else:
            pocket_features = data['pocket_one_hot'].to(self.device, FLOAT_TYPE)
            
        pocket = {
            'x': data['pocket_coords'].to(self.device, FLOAT_TYPE),
            'one_hot': pocket_features, # This now contains the concatenated features
            'size': data['num_pocket_nodes'].to(self.device, INT_TYPE),
            'mask': data['pocket_mask'].to(self.device, INT_TYPE)
        }
        return ligand, pocket

    def forward(self, data):
        ligand, pocket = self.get_ligand_and_pocket(data)

        delta_log_px, _, _, _, _, _, _, _, kl_prior, log_pN, t_int, xh_lig_hat, info, eps_t_lig, net_out_lig = \
            self.ddpm(ligand, pocket, return_info=True, return_loss_terms=True)

        if self.loss_type == 'l2':
            # --- THE WEIGHTED L2 LOSS FIX ---
            # Separate the error into coordinate and feature components
            error = eps_t_lig - net_out_lig
            coord_error = error[:, :self.x_dims]**2
            feat_error = error[:, self.x_dims:]**2

            # Apply the coordinate weight (you can tune this in your config)
            coord_loss_weight = self.hparams.get('coord_loss_weight', 1.0)
            weighted_coord_error = coord_loss_weight * coord_error

            # Recombine into a single error tensor
            squared_error = torch.cat([weighted_coord_error, feat_error], dim=1)
            # --- END FIX ---
            if self.virtual_nodes:
                squared_error[:, :self.x_dims] = torch.where(
                    ligand['one_hot'][:, self.virtual_atom:self.virtual_atom+1].bool(),
                    torch.zeros_like(squared_error[:, :self.x_dims]),
                    squared_error[:, :self.x_dims]
                )
            
            loss_t_per_sample = self.ddpm.sum_except_batch(squared_error, ligand['mask'])



            
            # Normalize by the number of dimensions (this is now a weighted normalization)
            num_feat_dims = self.atom_nf
            norm_factor = (coord_loss_weight * self.x_dims + num_feat_dims) * ligand['size']
            normalized_loss = loss_t_per_sample / (norm_factor + 1e-8)
            nll = normalized_loss
        else:
            raise NotImplementedError("Only L2 loss is supported in this stable version.")

        x_lig_hat = xh_lig_hat[:, :self.x_dims]
        h_lig_hat = xh_lig_hat[:, self.x_dims:]
            
        # 1. Repulsion Loss (from Lennard-Jones)
        if hasattr(self.hparams, 'auxiliary_loss') and self.hparams.auxiliary_loss and self.training:
            weighted_lj_repulsion = \
                self.auxiliary_weight_schedule(t_int.long()) * \
                self.lj_potential_repulsive_only(x_lig_hat, h_lig_hat, ligand['mask'])
            nll = nll + weighted_lj_repulsion
            info['weighted_lj'] = weighted_lj_repulsion.mean(0)
        
        # 2. Bond Attraction Loss (from Harmonic Potential)
        if hasattr(self.hparams, 'bond_params') and self.hparams.bond_params and self.hparams.bond_params.enabled and self.training:
            weight = self.hparams.bond_params.max_weight
            weighted_bond_potential = weight * self.harmonic_bond_potential(x_lig_hat, ligand['mask'])
            nll = nll + weighted_bond_potential
            info['bond_potential'] = weighted_bond_potential.mean(0)
        # --- END FIX ---

        # Fill info dict for logging
        info['error_t_lig'] = self.ddpm.sum_except_batch((eps_t_lig - net_out_lig) ** 2, ligand['mask']).mean()
        # ... (rest of info dict setup) ...
        return nll, info
    

    def lj_potential_repulsive_only(self, atom_x, atom_one_hot, batch_mask):
        adj = batch_mask[:, None] == batch_mask[None, :]
        adj = adj ^ torch.diag(torch.diag(adj))  # remove self-edges
        edges = torch.where(adj)

        # Compute pair-wise potentials
        dist = torch.sum((atom_x[edges[0]] - atom_x[edges[1]])**2, dim=1)
        r = torch.sqrt(dist + 1e-8) # Add epsilon for stability
        r = torch.clamp(r, min=0.1)

        # Get optimal radii from Lennard-Jones parameters
        lennard_jones_radii = torch.tensor(self.lj_rm, device=r.device) / 100.0
        lennard_jones_radii = lennard_jones_radii / self.ddpm.norm_values[0]
        atom_type_idx = atom_one_hot.argmax(1)

        # Clamp indices to be safe
        atom_type_idx = torch.clamp(atom_type_idx, 0, len(lennard_jones_radii) - 1)
            
        rm = lennard_jones_radii[atom_type_idx[edges[0]], atom_type_idx[edges[1]]]
        
        # Weeks-Chandler-Andersen (WCA) potential.
        # It is the LJ potential, shifted up, and set to 0 where it would be attractive.
        sigma = 2 ** (-1 / 6) * rm
        cutoff = 2**(1/6) * sigma # This is equal to rm
        
        energy = 4 * ((sigma / r) ** 12 - (sigma / r) ** 6) + 1 # Shifted LJ
        
        # Only apply the potential where r < cutoff (i.e., only the repulsive part)
        repulsive_only_energy = torch.where(r < cutoff, energy, torch.tensor(0.0, device=r.device))
        out = repulsive_only_energy
       
        out = scatter_add(out, edges[0], dim=0, dim_size=len(atom_x))
        
        # Clamp the maximum repulsion to prevent explosions from very close atoms
        #if self.clamp_lj is not None:
        #    out = torch.clamp(out, min=None, max=self.clamp_lj)
        
        return scatter_add(out, batch_mask, dim=0)
    
    def lj_potential(self, atom_x, atom_one_hot, batch_mask):
        adj = batch_mask[:, None] == batch_mask[None, :]
        adj = adj ^ torch.diag(torch.diag(adj))  # remove self-edges
        edges = torch.where(adj)

        # Compute pair-wise potentials
        dist = torch.sum((atom_x[edges[0]] - atom_x[edges[1]])**2, dim=1)
        # Add a small epsilon to the squared distance to prevent division by zero
        r = torch.sqrt(dist + 1e-6)
        r = torch.clamp(r, min=0.1) # Set minimum distance of 0.1 Angstrom
        
        # Get optimal radii
        lennard_jones_radii = torch.tensor(self.lj_rm, device=r.device)
        # unit conversion pm -> A
        lennard_jones_radii = lennard_jones_radii / 100.0
        # normalization
        lennard_jones_radii = lennard_jones_radii / self.ddpm.norm_values[0]
        atom_type_idx = atom_one_hot.argmax(1)
        
        # --- Check if indices are out of bounds
        if atom_type_idx.max() >= len(lennard_jones_radii) or atom_type_idx.min() < 0:
             print(f"Warning: atom_type_idx max {atom_type_idx.max()} out of bounds for lj_rm (size {len(lennard_jones_radii)})")
             # Clamp indices to be safe
             atom_type_idx = torch.clamp(atom_type_idx, 0, len(lennard_jones_radii) - 1)
             
        rm = lennard_jones_radii[atom_type_idx[edges[0]],
                                 atom_type_idx[edges[1]]]
        sigma = 2 ** (-1 / 6) * rm
        out = 4 * ((sigma / r) ** 12 - (sigma / r) ** 6)
       

        # Compute potential per atom
        out = scatter_add(out, edges[0], dim=0, dim_size=len(atom_x))

        #if self.clamp_lj is not None:
        #    out = torch.clamp(out, min=None, max=self.clamp_lj)
        
        # Sum potentials of all atoms
        return scatter_add(out, batch_mask, dim=0)
    
    
    def harmonic_bond_potential(self, atom_x, batch_mask):
        adj = batch_mask[:, None] == batch_mask[None, :]
        adj = adj ^ torch.diag(torch.diag(adj))  # remove self-edges
        edges = torch.where(adj)

        # Get distances in Angstrom space
        dist_sq = torch.sum((atom_x[edges[0]] - atom_x[edges[1]])**2, dim=1)
        r = torch.sqrt(dist_sq + 1e-8)

        # --- NEW: Re-introduce the cutoff ---
        # Only apply the potential to atom pairs within a physically
        # reasonable bonding distance. This prevents the global collapse.
        cutoff = self.hparams.bond_params.cutoff_radius
        mask = r < cutoff
        
        target_length = self.hparams.bond_params.target_length
        
        # Calculate potential only for pairs within the cutoff
        potential = torch.zeros_like(r)
        potential[mask] = (r[mask] - target_length)**2
        # --- END NEW ---

        out = potential
        
        # Safety clamp
        if hasattr(self.hparams.bond_params, 'clamp_b') and self.hparams.bond_params.clamp_b is not None:
            out = torch.clamp(out, min=None, max=self.hparams.bond_params.clamp_b)
    
        out = scatter_add(out, edges[0], dim=0, dim_size=len(atom_x))
        return scatter_add(out, batch_mask, dim=0)

    def log_metrics(self, metrics_dict, split, batch_size=None, **kwargs):
        for m, value in metrics_dict.items():
            self.log(f'{m}/{split}', value, batch_size=batch_size, **kwargs)

    def training_step(self, data, *args):

        # --- ROBUSTNESS FIX: Check for empty pockets before doing anything ---
        if 'num_pocket_nodes' in data and (data['num_pocket_nodes'] == 0).any():
            print("WARNING: Skipping batch because it contains a pocket with zero atoms.")
            return None

        # STABILITY FIX: Add jitter to prevent coincident atoms
        if self.augment_noise > 0:
            noise = self.augment_noise * torch.randn_like(data['lig_coords'])
            data['lig_coords'] = data['lig_coords'] + noise
            if 'pocket_coords' in data:
                p_noise = self.augment_noise * torch.randn_like(data['pocket_coords'])
                data['pocket_coords'] = data['pocket_coords'] + p_noise

        # STABILITY FIX: Check for NaN in input data
        if not torch.isfinite(data['lig_coords']).all() or \
           ('pocket_coords' in data and not torch.isfinite(data['pocket_coords']).all()):
            print("WARNING: NaN detected in input coordinates, skipping batch.")
            return None

        try:
            nll, info = self.forward(data)
        except RuntimeError as e:
            if 'out of memory' in str(e) or 'CUDA' in str(e):
                print(f'WARNING: Caught GPU error: {e}. Skipping batch.')
                torch.cuda.empty_cache()
                return None
            else:
                raise e

        # STABILITY FIX: Check for NaN in loss
        if not torch.isfinite(nll).all():
            print("WARNING: NaN loss detected. Skipping batch.")
            return None

        loss = nll.mean(0)
        info['loss'] = loss
        self.log(
            'loss/train', 
            loss, 
            prog_bar=True, 
            batch_size=len(data['num_lig_atoms'])
        )
        info_to_log = info.copy()
        if 'loss' in info_to_log:
            del info_to_log['loss']
        # --- END NEW FIX ---

        # The line below logs all other metrics ('weighted_lj', 'error_t_lig', etc.)
        self.log_metrics(info_to_log, 'train', batch_size=len(data['num_lig_atoms']))
        return info
    
    def _shared_eval(self, data, prefix, *args):
        # STABILITY FIX: Always apply a small jitter during eval
        jitter_val = 1e-4
        if 'lig_coords' in data:
            data['lig_coords'] += jitter_val * torch.randn_like(data['lig_coords'])
        if 'pocket_coords' in data:
            data['pocket_coords'] += jitter_val * torch.randn_like(data['pocket_coords'])
        
        try:
            nll, info = self.forward(data)
        except RuntimeError as e:
            if 'out of memory' in str(e) or 'CUDA' in str(e):
                print(f'WARNING: GPU error during {prefix}, skipping batch')
                torch.cuda.empty_cache()
                return {'loss': torch.tensor(float('nan'))} # Return NaN loss to be handled
            else:
                raise e
        
        if not torch.isfinite(nll).all():
            print(f"WARNING: NaN in {prefix} loss. Skipping logging.")
            return {'loss': torch.tensor(float('nan'))}
        
        loss = nll.mean(0)
        info['loss'] = loss
        self.log(
            f'loss/{prefix}', 
            loss, 
            prog_bar=True, 
            batch_size=len(data['num_lig_atoms']), 
            sync_dist=True
            
        )
        info_to_log = info.copy()
        if 'loss' in info_to_log:
            del info_to_log['loss']
        # Log all other metrics without the progress bar
        self.log_metrics(info_to_log, 'train', batch_size=len(data['num_lig_atoms']))

        return info
    def validation_step(self, data, *args):
        self._shared_eval(data, 'val', *args)

    def test_step(self, data, *args):
        self._shared_eval(data, 'test', *args)
    
    
    def on_before_optimizer_step(self, optimizer):
        # --- NEW: Manual LR Warmup Logic ---
        if self.trainer.global_step < self.hparams.warmup_steps:
            # Calculate the learning rate scale (e.g., step 1/10, 2/10, ...)
            lr_scale = float(self.trainer.global_step + 1) / float(self.hparams.warmup_steps)
            
            # Manually set the learning rate for this step
            base_lr = self.hparams.lr 
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr * lr_scale
        
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        if total_norm > 100.0:
            print(f"WARNING: Large gradient norm detected: {total_norm:.2f}")
        
        self.log('grad_norm', total_norm, prog_bar=True)

    def on_validation_epoch_end(self):

        # Perform validation on single GPU
        if not self.trainer.is_global_zero:
            return

        suffix = '' if self.mode == 'joint' else '_given_pocket'

        if (self.current_epoch + 1) % self.eval_epochs == 0:
            tic = time()

            sampling_results = getattr(self, 'sample_and_analyze' + suffix)(
                self.eval_params.n_eval_samples, self.val_dataset,
                batch_size=self.eval_batch_size)
            self.log_metrics(sampling_results, 'val')

            print(f'Evaluation took {time() - tic:.2f} seconds')

        if (self.current_epoch + 1) % self.visualize_sample_epoch == 0:
            tic = time()
            getattr(self, 'sample_chain_and_save' + suffix)(
                self.eval_params.n_visualize_samples)
            print(f'Sample visualization took {time() - tic:.2f} seconds')

        if (self.current_epoch + 1) % self.visualize_chain_epoch == 0:
            tic = time()
            getattr(self, 'sample_chain_and_save' + suffix)(
                self.eval_params.keep_frames)
            print(f'Chain visualization took {time() - tic:.2f} seconds')

    @torch.no_grad()
    def sample_and_analyze(self, n_samples, dataset=None, batch_size=None):
        print(f'Analyzing sampled molecules at epoch {self.current_epoch}...')

        batch_size = self.batch_size if batch_size is None else batch_size
        batch_size = min(batch_size, n_samples)

        # each item in molecules is a tuple (position, atom_type_encoded)
        molecules = []
        atom_types = []
        aa_types = []
        for i in range(math.ceil(n_samples / batch_size)):

            n_samples_batch = min(batch_size, n_samples - len(molecules))

            num_nodes_lig, num_nodes_pocket = \
                self.ddpm.size_distribution.sample(n_samples_batch)

            xh_lig, xh_pocket, lig_mask, _ = self.ddpm.sample(
                n_samples_batch, num_nodes_lig, num_nodes_pocket,
                device=self.device)

            x = xh_lig[:, :self.x_dims].detach().cpu()
            atom_type = xh_lig[:, self.x_dims:].argmax(1).detach().cpu()
            lig_mask = lig_mask.cpu()

            molecules.extend(list(
                zip(utils.batch_to_list(x, lig_mask),
                    utils.batch_to_list(atom_type, lig_mask))
            ))

            atom_types.extend(atom_type.tolist())
            if self.pocket_representation != 'atomica':
                aa_types.extend(
                    xh_pocket[:, self.x_dims:].argmax(1).detach().cpu().tolist())

        return self.analyze_sample(molecules, atom_types, aa_types)

    def analyze_sample(self, molecules, atom_types, aa_types, receptors=None):
        # Distribution of node types
        kl_div_atom = self.ligand_type_distribution.kl_divergence(atom_types) \
            if self.ligand_type_distribution is not None else -1
        
        # --- calc KL for embeddings ---
        kl_div_aa = -1
        if self.pocket_representation == 'CA':
             kl_div_aa = self.pocket_type_distribution.kl_divergence(aa_types) \
                if self.pocket_type_distribution is not None else -1

        # Convert into rdmols
        rdmols = []
        for i, graph in enumerate(molecules):
            mol = build_molecule(*graph, self.dataset_info)
            if mol is None:
                # RDKit/OpenBabel failed, likely due to NaN
                print(f"Warning: Skipping molecule {i} in analysis, build_molecule returned None.")
                continue
            rdmols.append(mol)

        # Other basic metrics
        (validity, connectivity, uniqueness, novelty), (_, connected_mols) = \
            self.ligand_metrics.evaluate_rdmols(rdmols)

        qed, sa, logp, lipinski, diversity = \
            self.molecule_properties.evaluate_mean(connected_mols)

        out = {
            'kl_div_atom_types': kl_div_atom,
            'kl_div_residue_types': kl_div_aa,
            'Validity': validity,
            'Connectivity': connectivity,
            'Uniqueness': uniqueness,
            'Novelty': novelty,
            'QED': qed,
            'SA': sa,
            'LogP': logp,
            'Lipinski': lipinski,
            'Diversity': diversity
        }

        # Simple docking score
        if receptors is not None:
            # out['smina_score'] = np.mean(smina_score(rdmols, receptors))
            out['smina_score'] = np.mean(smina_score(connected_mols, receptors))

        return out

    def get_full_path(self, receptor_name):
        # --- Handle .pt files from 'atomica' dataset ---
        if self.pocket_representation == 'atomica':
           
            return None # Path(self.datadir, 'val', receptor_name)
            
        pdb, suffix = receptor_name.split('.')
        receptor_name = f'{pdb.upper()}-{suffix}.pdb'
        return Path(self.datadir, 'val', receptor_name)

    @torch.no_grad()
    def sample_and_analyze_given_pocket(self, n_samples, dataset=None,
                                        batch_size=None):
        print(f'Analyzing sampled molecules given pockets at epoch '
              f'{self.current_epoch}...')

        batch_size = self.batch_size if batch_size is None else batch_size
        batch_size = min(batch_size, n_samples)

        # each item in molecules is a tuple (position, atom_type_encoded)
        molecules = []
        atom_types = []
        aa_types = []
        receptors = []
        for i in range(math.ceil(n_samples / batch_size)):

            n_samples_batch = min(batch_size, n_samples - len(molecules))

            # Create a batch
            batch = dataset.collate_fn(
                [dataset[(i * batch_size + j) % len(dataset)]
                 for j in range(n_samples_batch)]
            )

            ligand, pocket = self.get_ligand_and_pocket(batch)
            
            # --- Handle 'atomica' dataset names
            if self.pocket_representation == 'atomica':
                # 'names' field from .pt file
                receptors.extend(batch['name']) 
            else:
                receptors.extend([self.get_full_path(x) for x in batch['receptors']])

            if self.virtual_nodes:
                num_nodes_lig = self.max_num_nodes
            else:
                n_samples_in_batch = len(pocket['size'])
                # Sample pairs of (ligand, pocket) node counts
                num_nodes_pairs = self.ddpm.size_distribution.sample_conditional(
                    n1=None, n2=pocket['size'], n_samples=n_samples_in_batch)
                # --> Select only the ligand node counts (the first column) <--
                num_nodes_lig = num_nodes_pairs[:, 0]

            xh_lig, xh_pocket, lig_mask, _ = self.ddpm.sample_given_pocket(
                pocket, num_nodes_lig)

            x = xh_lig[:, :self.x_dims].detach().cpu()
            atom_type = xh_lig[:, self.x_dims:].argmax(1).detach().cpu()
            lig_mask = lig_mask.cpu()

            if self.virtual_nodes:
                # Remove virtual nodes for analysis
                vnode_mask = (atom_type == self.virtual_atom)
                x = x[~vnode_mask, :]
                atom_type = atom_type[~vnode_mask]
                lig_mask = lig_mask[~vnode_mask]

            molecules.extend(list(
                zip(utils.batch_to_list(x, lig_mask),
                    utils.batch_to_list(atom_type, lig_mask))
            ))

            atom_types.extend(atom_type.tolist())
            if self.pocket_representation != 'atomica':
                aa_types.extend(
                    xh_pocket[:, self.x_dims:].argmax(1).detach().cpu().tolist())
        
        # --- Disable docking for atomica unless PDB paths are stored
        if self.pocket_representation == 'atomica':
             print("Docking analysis skipped for 'atomica' representation.")
             receptors = None

        return self.analyze_sample(molecules, atom_types, aa_types,
                                   receptors=receptors)

    

    def sample_and_save(self, n_samples):
        num_nodes_lig, num_nodes_pocket = \
            self.ddpm.size_distribution.sample(n_samples)

        xh_lig, xh_pocket, lig_mask, pocket_mask = \
            self.ddpm.sample(n_samples, num_nodes_lig, num_nodes_pocket,
                             device=self.device)
        
        # --- PREPARE LIGAND DATA ---
        x_lig = xh_lig[:, :self.x_dims]
        one_hot_lig = xh_lig[:, self.x_dims:]
        atom_type_lig = torch.argmax(one_hot_lig, dim=1)

        # --- PREPARE POCKET DATA FOR VISUALIZATION ---
        if self.pocket_representation == 'CA':
            x_pocket, one_hot_pocket = utils.residues_to_atoms(
                xh_pocket[:, :self.x_dims], self.lig_type_encoder)
        elif self.pocket_representation == 'atomica':
            x_pocket = xh_pocket[:, :self.x_dims]
            one_hot_pocket = torch.zeros(len(x_pocket), len(self.lig_type_decoder), device=self.device)
            other_idx = self.lig_type_encoder.get('others', 0)
            one_hot_pocket[:, other_idx] = 1
        else: # 'full-atom'
            x_pocket, one_hot_pocket = \
                xh_pocket[:, :self.x_dims], xh_pocket[:, self.x_dims:]
        atom_type_pocket = torch.argmax(one_hot_pocket, dim=1)
        
        outdir = Path(self.outdir, f'epoch_{self.current_epoch}')
        
        # --- SAVE LIGAND ONLY to .xyz file (this is already correct) ---
        save_xyz_file(str(outdir) + '/', one_hot_lig, x_lig, self.lig_type_decoder,
                      name='molecule', batch_mask=lig_mask)

        # --- FIX START: VISUALIZE EACH SAMPLE IN THE BATCH ---
        # Determine the number of samples from the batch masks
        num_samples_in_batch = lig_mask.max().item() + 1
        for i in range(num_samples_in_batch):
            # Get masks for the current sample
            lig_indices = (lig_mask == i)
            pocket_indices = (pocket_mask == i)
            
            # Define a unique path for the image
            save_path = str(outdir / f'molecule_{i:03d}_with_pocket.png')

            # Check if there are any ligand atoms to plot for this sample
            if lig_indices.sum() > 0:
                plot_molecule_and_pocket(
                    positions_lig=x_lig[lig_indices].cpu(),
                    atom_type_lig=atom_type_lig[lig_indices].cpu().numpy(),
                    positions_pocket=x_pocket[pocket_indices].cpu(),
                    atom_type_pocket=atom_type_pocket[pocket_indices].cpu().numpy(),
                    dataset_info=self.dataset_info,
                    save_path=save_path
                )
        # --- FIX END ---

    def sample_chain_and_save_given_pocket(self, n_samples):
        # ... (function setup remains the same) ...
        batch = self.val_dataset.collate_fn(
            [self.val_dataset[i] for i in torch.randint(len(self.val_dataset),
                                                        size=(n_samples,))]
        )
        ligand, pocket = self.get_ligand_and_pocket(batch)

        if self.virtual_nodes:
            num_nodes_lig = self.max_num_nodes
        else:
            n_samples_in_batch = len(pocket['size'])
            num_nodes_pairs = self.ddpm.size_distribution.sample_conditional(
                n1=None, n2=pocket['size'], n_samples=n_samples_in_batch)
            num_nodes_lig = num_nodes_pairs[:, 0]

        xh_lig, xh_pocket, lig_mask, pocket_mask = \
            self.ddpm.sample_given_pocket(pocket, num_nodes_lig)
        
        # --- PREPARE LIGAND DATA ---
        x_lig = xh_lig[:, :self.x_dims]
        one_hot_lig = xh_lig[:, self.x_dims:]
        atom_type_lig = torch.argmax(one_hot_lig, dim=1)

        # --- PREPARE POCKET DATA FOR VISUALIZATION ---
        if self.pocket_representation == 'CA':
            x_pocket, one_hot_pocket = utils.residues_to_atoms(
                xh_pocket[:, :self.x_dims], self.lig_type_encoder)
        elif self.pocket_representation == 'atomica':
            x_pocket = xh_pocket[:, :self.x_dims]
            one_hot_pocket = torch.zeros(len(x_pocket), len(self.lig_type_decoder), device=self.device)
            other_idx = self.lig_type_encoder.get('others', 0)
            one_hot_pocket[:, other_idx] = 1
        else: # 'full-atom'
            x_pocket, one_hot_pocket = \
                xh_pocket[:, :self.x_dims], xh_pocket[:, self.x_dims:]
        atom_type_pocket = torch.argmax(one_hot_pocket, dim=1)

        outdir = Path(self.outdir, f'epoch_{self.current_epoch}')
        
        # --- SAVE LIGAND ONLY to .xyz file (this is already correct) ---
        save_xyz_file(str(outdir) + '/', one_hot_lig, x_lig, self.lig_type_decoder,
                      name='molecule', batch_mask=lig_mask)

        # --- FIX START: VISUALIZE EACH SAMPLE IN THE BATCH ---
        # Determine the number of samples from the batch masks
        num_samples_in_batch = lig_mask.max().item() + 1
        for i in range(num_samples_in_batch):
            # Get masks for the current sample
            lig_indices = (lig_mask == i)
            pocket_indices = (pocket_mask == i)
            
            # Define a unique path for the image
            save_path = str(outdir / f'molecule_{i:03d}_with_pocket.png')

            # Check if there are any ligand atoms to plot for this sample
            if lig_indices.sum() > 0:
                plot_molecule_and_pocket(
                    positions_lig=x_lig[lig_indices].cpu(),
                    atom_type_lig=atom_type_lig[lig_indices].cpu().numpy(),
                    positions_pocket=x_pocket[pocket_indices].cpu(),
                    atom_type_pocket=atom_type_pocket[pocket_indices].cpu().numpy(),
                    dataset_info=self.dataset_info,
                    save_path=save_path
                )
        # --- FIX END ---

    def _load_atomica_model_if_needed(self):
        if not ATOMICA_IMPORTS_OK:
             raise ImportError("ATOMICA modules not found. Cannot run inference.")
        if self.atomica_model is None:
            print("Loading ATOMICA model for inference...")
            atomica_args = Namespace(
                model_ckpt=self.hparams.atomica_model_path,
                model_config=self.hparams.atomica_model_config,
                model_weights=self.hparams.atomica_model_weights
            )
            if atomica_args.model_ckpt is None and \
               (atomica_args.model_config is None or atomica_args.model_weights is None):
               raise ValueError("Must provide 'atomica_model_path' (ckpt) or "
                                "'atomica_model_config' and 'atomica_model_weights' "
                                "in config for inference.")
                                
            self.atomica_model = load_atomica_model(atomica_args).to(self.device).eval()
            print("ATOMICA model loaded.")

    # --- : prepare_pocket for Phase 4 Inference ---
    def prepare_pocket(self, biopython_residues, repeats=1):

        if self.pocket_representation != 'atomica':
            # --- for 'CA' or 'full-atom' ---
            if self.pocket_representation == 'CA':
                pocket_coord = torch.tensor(np.array(
                    [res['CA'].get_coord() for res in biopython_residues]),
                    device=self.device, dtype=FLOAT_TYPE)
                pocket_types = torch.tensor(
                    [self.pocket_type_encoder[PROTEIN_LETTERS_3TO1.get(res.get_resname().upper(), "X")]
                     for res in biopython_residues], device=self.device)
            else: # 'full-atom'
                pocket_atoms = [a for res in biopython_residues
                                for a in res.get_atoms()
                                if (a.element.capitalize() in self.pocket_type_encoder or a.element != 'H')]
                pocket_coord = torch.tensor(np.array(
                    [a.get_coord() for a in pocket_atoms]),
                    device=self.device, dtype=FLOAT_TYPE)
                pocket_types = torch.tensor(
                    [self.pocket_type_encoder[a.element.capitalize()]
                     for a in pocket_atoms], device=self.device)

            pocket_one_hot = F.one_hot(
                pocket_types, num_classes=len(self.pocket_type_encoder)
            )

            pocket_size = torch.tensor([len(pocket_coord)] * repeats,
                                       device=self.device, dtype=INT_TYPE)
            pocket_mask = torch.repeat_interleave(
                torch.arange(repeats, device=self.device, dtype=INT_TYPE),
                len(pocket_coord)
            )

            pocket = {
                'x': pocket_coord.repeat(repeats, 1),
                'one_hot': pocket_one_hot.repeat(repeats, 1),
                'size': pocket_size,
                'mask': pocket_mask
            }
            return pocket

        
        

    def generate_ligands(self, pdb_file, n_samples, pocket_ids=None,
                         ref_ligand=None, num_nodes_lig=None, sanitize=False,
                         largest_frag=False, relax_iter=0, timesteps=None,
                         n_nodes_bias=0, n_nodes_min=0, **kwargs):
        """
        Generate ligands given a pocket
        Args:
            pdb_file: PDB filename
            n_samples: number of samples
            pocket_ids: list of pocket residues in <chain>:<resi> format
            ref_ligand: alternative way of defining the pocket based on a
                reference ligand given in <chain>:<resi> format if the ligand is
                contained in the PDB file, or path to an SDF file that
                contains the ligand
            num_nodes_lig: number of ligand nodes for each sample (list of
                integers), sampled randomly if 'None'
            sanitize: whether to sanitize molecules or not
            largest_frag: only return the largest fragment
            relax_iter: number of force field optimization steps
            timesteps: number of denoising steps, use training value if None
            n_nodes_bias: added to the sampled (or provided) number of nodes
            n_nodes_min: lower bound on the number of sampled nodes
            kwargs: additional inpainting parameters
        Returns:
            list of molecules
        """

        assert (pocket_ids is None) ^ (ref_ligand is None)

        self.ddpm.eval()

        # Load PDB
        pdb_struct = PDBParser(QUIET=True).get_structure('', pdb_file)[0]
        if pocket_ids is not None:
            # define pocket with list of residues
            residues = [
                pdb_struct[x.split(':')[0]][(' ', int(x.split(':')[1]), ' ')]
                for x in pocket_ids]

        else:
            # define pocket with reference ligand
            residues = utils.get_pocket_from_ligand(pdb_struct, ref_ligand)

        
        pocket = self.prepare_pocket(residues, repeats=n_samples)

        # Pocket's center of mass
        pocket_com_before = scatter_mean(pocket['x'], pocket['mask'], dim=0)

        # Create dummy ligands
        if num_nodes_lig is None:
            num_nodes_lig = self.ddpm.size_distribution.sample_conditional(
                n1=None, n2=pocket['size'])

        # Add bias
        num_nodes_lig = num_nodes_lig + n_nodes_bias

        # Apply minimum ligand size
        num_nodes_lig = torch.clamp(num_nodes_lig, min=n_nodes_min)

        # Use inpainting
        if type(self.ddpm) == EnVariationalDiffusion:
            lig_mask = utils.num_nodes_to_batch_mask(
                len(num_nodes_lig), num_nodes_lig, self.device)

            ligand = {
                'x': torch.zeros((len(lig_mask), self.x_dims),
                                 device=self.device, dtype=FLOAT_TYPE),
                'one_hot': torch.zeros((len(lig_mask), self.atom_nf),
                                       device=self.device, dtype=FLOAT_TYPE),
                'size': num_nodes_lig,
                'mask': lig_mask
            }

            # Fix all pocket nodes but sample
            lig_mask_fixed = torch.zeros(len(lig_mask), device=self.device)
            pocket_mask_fixed = torch.ones(len(pocket['mask']),
                                           device=self.device)

            xh_lig, xh_pocket, lig_mask, pocket_mask = self.ddpm.inpaint(
                ligand, pocket, lig_mask_fixed, pocket_mask_fixed,
                timesteps=timesteps, **kwargs)

        # Use conditional generation
        elif type(self.ddpm) == ConditionalDDPM:
            xh_lig, xh_pocket, lig_mask, pocket_mask = \
                self.ddpm.sample_given_pocket(pocket, num_nodes_lig,
                                              timesteps=timesteps)

        else:
            raise NotImplementedError

        # Move generated molecule back to the original pocket position
        pocket_com_after = scatter_mean(
            xh_pocket[:, :self.x_dims], pocket_mask, dim=0)

        xh_pocket[:, :self.x_dims] += \
            (pocket_com_before - pocket_com_after)[pocket_mask]
        xh_lig[:, :self.x_dims] += \
            (pocket_com_before - pocket_com_after)[lig_mask]

        # Build mol objects
        x = xh_lig[:, :self.x_dims].detach().cpu()
        atom_type = xh_lig[:, self.x_dims:].argmax(1).detach().cpu()
        lig_mask = lig_mask.cpu()

        molecules = []
        for mol_pc in zip(utils.batch_to_list(x, lig_mask),
                          utils.batch_to_list(atom_type, lig_mask)):

            mol = build_molecule(*mol_pc, self.dataset_info, add_coords=True)
            mol = process_molecule(mol,
                                   add_hydrogens=False,
                                   sanitize=sanitize,
                                   relax_iter=relax_iter,
                                   largest_frag=largest_frag)
            if mol is not None:
                molecules.append(mol)

        return molecules




class WeightSchedule:
    def __init__(self, T, max_weight, mode='linear'):
        if mode == 'linear':
            self.weights = torch.linspace(max_weight, 0, T + 1)
        elif mode == 'constant':
            self.weights = max_weight * torch.ones(T + 1)
        else:
            raise NotImplementedError(f'{mode} weight schedule is not '
                                      f'available.')

    def __call__(self, t_array):
        """ all values in t_array are assumed to be integers in [0, T] """
        return self.weights.to(t_array.device)[t_array]

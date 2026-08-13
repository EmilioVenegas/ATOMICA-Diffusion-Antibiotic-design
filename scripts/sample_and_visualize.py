import sys
import os
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Anchor on the repo root rather than the working directory, so these run from
# anywhere. First entry resolves `ATOMICA.*`; second resolves DiffSBDD's own
# top-level modules (utils, lightning_modules, analysis).
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'DiffSBDD'))

from lightning_modules import LigandPocketDDPM
from analysis.visualization import plot_molecule_and_pocket

def calculate_clashes(lig_coords, pocket_coords, lig_atom_types, pocket_atom_types, 
                     dataset_info, tolerance=0.6):
    """
    Calculate clashes between ligand and pocket.
    Tolerance: Fraction of VDW sum. < 1.0 means overlap allowed.
               0.6 is a very strict clash (severe overlap).
    """
    # Simple VDW radii (approximate)
    vdw_radii = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'P': 1.80, 
                 'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98, 'H': 1.20,
                 'X': 1.70} # Default for unknown
    
    atom_decoder = dataset_info['atom_decoder']
    
    clashes = []
    min_dists = []
    
    for i, l_pos in enumerate(lig_coords):
        l_type = atom_decoder[lig_atom_types[i]]
        l_rad = vdw_radii.get(l_type, 1.7)
        
        # Vectorized distance to all pocket atoms
        dists = torch.norm(pocket_coords - l_pos, dim=1)
        min_dist = torch.min(dists).item()
        min_dists.append(min_dist)
        
        # Check clashes
        # We don't know pocket atom types easily in atomica mode (all 'X')
        # So assume Carbon radius for pocket atoms
        p_rad = 1.7 
        
        threshold = (l_rad + p_rad) * tolerance
        
        if min_dist < threshold:
            clashes.append((i, min_dist, threshold))
            
    return clashes, min_dists

def sample_and_visualize(checkpoint_path, output_dir, n_samples=5, corrector_steps=0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model from {checkpoint_path}...")
    
    # Load model
    model = LigandPocketDDPM.load_from_checkpoint(checkpoint_path, map_location=device)
    model = model.to(device)
    model.eval()
    
    # Setup output
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Load validation set
    print("Loading validation dataset...")
    model.setup(stage='fit') # Loads train and val
    val_loader = model.val_dataloader()
    
    print(f"Generating {n_samples} samples...")
    
    count = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if count >= n_samples:
                break
                
            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            ligand, pocket = model.get_ligand_and_pocket(batch)
            
            # Generate sample
            # We use the pocket size from the batch
            num_nodes_lig = model.ddpm.size_distribution.sample_conditional(
                    n1=None, n2=pocket['size'], n_samples=1)[:, 0]
            
            # Sample
            print(f"Sampling batch with corrector_steps={corrector_steps}...")
            xh_lig, xh_pocket, lig_mask, _ = model.ddpm.sample_given_pocket(
                pocket, num_nodes_lig, corrector_steps=corrector_steps)
            
            # Process batch
            batch_size = len(pocket['size'])
            
            for i in range(batch_size):
                if count >= n_samples:
                    break
                
                # Extract single sample
                l_mask = (lig_mask == i)
                p_mask = (pocket['mask'] == i)
                
                if l_mask.sum() == 0 or p_mask.sum() == 0:
                    continue
                
                l_pos = xh_lig[l_mask, :3]
                l_one_hot = xh_lig[l_mask, 3:]
                l_types = torch.argmax(l_one_hot, dim=1)
                
                p_pos = xh_pocket[p_mask, :3]
                # Pocket types are hidden in atomica mode, assume generic
                p_types = torch.zeros(len(p_pos), dtype=torch.long) 
                
                # Calculate Clashes
                clashes, min_dists = calculate_clashes(
                    l_pos, p_pos, l_types, p_types, model.dataset_info
                )
                
                # Calculate Centroid Distance
                l_center = l_pos.mean(dim=0)
                p_center = p_pos.mean(dim=0)
                center_dist = torch.norm(l_center - p_center).item()
                
                print(f"Sample {count}: {len(clashes)} clashes, Center Dist: {center_dist:.2f} A")
                print(f"  Ligand Coords: Min {l_pos.min():.2f}, Max {l_pos.max():.2f}, Std {l_pos.std():.2f}")
                print(f"  Pocket Coords: Min {p_pos.min():.2f}, Max {p_pos.max():.2f}, Std {p_pos.std():.2f}")
                
                # --- VISUALIZATION ---
                # Plot ligand and pocket
                save_file = out_path / f"sample_{count:03d}_clash_{len(clashes)}.png"
                
                # We need to pass CPU tensors to the plot function
                plot_molecule_and_pocket(
                    l_pos.cpu(), l_types.cpu(), 
                    p_pos.cpu(), p_types.cpu(), 
                    model.dataset_info, 
                    save_path=str(save_file),
                    spheres_3d=False
                )
                print(f"Saved plot to {save_file}")
                
                # Save XYZ as well for reference
                xyz_file = out_path / f"sample_{count:03d}_clash_{len(clashes)}.xyz"
                with open(xyz_file, 'w') as f:
                    n_atoms = len(l_pos) + len(p_pos)
                    f.write(f"{n_atoms}\n")
                    f.write(f"Generated sample {count}. Clashes: {len(clashes)}. Center Dist: {center_dist:.2f}\n")
                    
                    atom_decoder = model.dataset_info['atom_decoder']
                    for j, pos in enumerate(l_pos):
                        atom = atom_decoder[l_types[j]]
                        f.write(f"{atom} {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}\n")
                        
                    for j, pos in enumerate(p_pos):
                        f.write(f"N {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}\n")

                count += 1

if __name__ == "__main__":
    # Use the checkpoint from the user's previous context or a default
    # The user mentioned "my_logs/tanh-active-context-geom-v1/checkpoints/last-v3.ckpt" in verify_sampling.py
    # I'll use that as default but allow override via args if I were adding argparse, 
    # but for this script I'll keep it simple as requested.
    
    ckpt = "my_logs/tanh-active-context-geom-v1/checkpoints/last-v3.ckpt"
    out = "visualizations"
    
    # Check if checkpoint exists, if not try to find one
    if not os.path.exists(ckpt):
        print(f"Checkpoint {ckpt} not found. Searching for others...")
        # Try to find any .ckpt in my_logs
        found_ckpts = list(Path("my_logs").rglob("*.ckpt"))
        if found_ckpts:
            ckpt = str(found_ckpts[0])
            print(f"Using found checkpoint: {ckpt}")
        else:
            print("No checkpoints found!")
            sys.exit(1)
            
    sample_and_visualize(ckpt, out, n_samples=5, corrector_steps=5)

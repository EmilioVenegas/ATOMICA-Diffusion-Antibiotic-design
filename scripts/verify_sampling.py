import sys
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Anchor on the repo root rather than the working directory, so these run from
# anywhere. First entry resolves `ATOMICA.*`; second resolves DiffSBDD's own
# top-level modules (utils, lightning_modules, analysis).
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'DiffSBDD'))

from lightning_modules import LigandPocketDDPM
from analysis.visualization import save_xyz_file
from constants import dataset_params

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

def verify_sampling(checkpoint_path, output_dir, n_samples=20, corrector_steps=0):
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
    
    results = []
    
    with torch.no_grad():
        count = 0
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
                
                # Save Combined XYZ
                filename = f"sample_{count:03d}_clash_{len(clashes)}.xyz"
                filepath = out_path / filename
                
                with open(filepath, 'w') as f:
                    n_atoms = len(l_pos) + len(p_pos)
                    f.write(f"{n_atoms}\n")
                    f.write(f"Generated sample {count}. Clashes: {len(clashes)}. Center Dist: {center_dist:.2f}\n")
                    
                    # Write Ligand
                    atom_decoder = model.dataset_info['atom_decoder']
                    for j, pos in enumerate(l_pos):
                        atom = atom_decoder[l_types[j]]
                        f.write(f"{atom} {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}\n")
                        
                    # Write Pocket (as 'N' ...)
                    # Using 'N' (Nitrogen) for pocket to distinguish from Ligand (mostly C)
                    # Or use 'Ca' if supported
                    # Update: Decode actual atom types if available
                    p_one_hot = xh_pocket[p_mask, 3:]
                    p_types_idx = torch.argmax(p_one_hot, dim=1)
                    
                    # In atomica mode, pocket one-hot might be amino acids OR atoms
                    # If dim is small (e.g. 9), it's atoms. If large (20), it's residues.
                    # Config says atomica_one_hot_dim: 9. So it's atoms!
                    # We can use the same decoder.
                    
                    for j, pos in enumerate(p_pos):
                        try:
                            atom = atom_decoder[p_types_idx[j].item()]
                        except Exception as e:
                            print(f"Error decoding atom: {e}")
                            atom = 'N' # Fallback
                        f.write(f"{atom} {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}\n")
                
                results.append({
                    'id': count,
                    'clashes': len(clashes),
                    'min_dist': min(min_dists) if min_dists else 0,
                    'center_dist': center_dist,
                    'n_atoms': len(l_pos)
                })
                
                count += 1
                
    # Print Summary
    print("\n" + "="*50)
    print("SAMPLING VERIFICATION RESULTS")
    print("="*50)
    
    avg_clashes = np.mean([r['clashes'] for r in results])
    avg_dist = np.mean([r['center_dist'] for r in results])
    
    print(f"Analyzed {len(results)} samples")
    print(f"Average Severe Clashes (<0.6*VDW): {avg_clashes:.2f}")
    print(f"Average Ligand-Pocket Center Dist: {avg_dist:.2f} Å")
    
    print("\nDetailed Samples:")
    print(f"{'ID':<5} {'Clashes':<10} {'Min Dist':<10} {'Center Dist':<15} {'Status'}")
    print("-" * 60)
    
    for r in results:
        status = "✅ OK"
        if r['clashes'] > 2: status = "⚠️ Clashes"
        if r['center_dist'] > 10: status = "⚠️ Far"
        
        print(f"{r['id']:<5} {r['clashes']:<10} {r['min_dist']:.2f} Å     {r['center_dist']:.2f} Å        {status}")
        
    print("\n" + "="*50)
    print(f"Saved combined XYZ files to {output_dir}")
    print("Open these files in PyMOL/Chimera to verify geometry.")
    print("Pocket atoms are labeled as 'N' for visualization.")

if __name__ == "__main__":
    ckpt = "my_logs/tanh-active-context-geom-v1/checkpoints/last-v3.ckpt"
    out = "verification_results"
    verify_sampling(ckpt, out, corrector_steps=10)

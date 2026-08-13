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

from analysis.molecule_builder import build_molecule
from analysis.metrics import rdmol_to_smiles
from constants import dataset_params

def extract_smiles(data_dir, output_file):
    """
    Extract SMILES from processed training data for metric calculation.
    """
    data_path = Path(data_dir)
    files = list(data_path.glob('*.pt'))
    
    if not files:
        print(f"No .pt files found in {data_dir}")
        return

    print(f"Found {len(files)} training files. Extracting SMILES...")
    
    # Get dataset info for decoding
    dataset_info = dataset_params['atomica_PL']
    
    smiles_list = []
    valid_count = 0
    
    for f in tqdm(files):
        try:
            data = torch.load(f)
            
            # Extract ligand data
            lig_coords = data['lig_coords']
            lig_one_hot = data['lig_one_hot']
            atom_types = torch.argmax(lig_one_hot, dim=1)
            
            # Build molecule
            mol = build_molecule(lig_coords, atom_types, dataset_info)
            
            if mol:
                smiles = rdmol_to_smiles(mol)
                if smiles:
                    smiles_list.append(smiles)
                    valid_count += 1
                    
        except Exception as e:
            print(f"Error processing {f}: {e}")
            continue
            
    # Save as numpy array
    np.save(output_file, np.array(smiles_list))
    
    print(f"\nSuccessfully extracted {len(smiles_list)} SMILES.")
    print(f"Saved to {output_file}")
    print(f"Validity rate: {100 * valid_count / len(files):.1f}%")

if __name__ == "__main__":
    # Adjust these paths if needed
    TRAIN_DIR = "data/processed_crossdocked_atomica/train"
    OUTPUT_FILE = "data/crossdocked_smiles.npy"
    
    extract_smiles(TRAIN_DIR, OUTPUT_FILE)

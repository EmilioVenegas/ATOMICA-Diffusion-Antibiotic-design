import argparse
import gzip
import json
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm



def format_atomica_batch(pocket_coords, pocket_atom_types, pocket_block_types, device):
    """
    Formats the raw pocket data into the batch dictionary
    expected by the ATOMICA model's .infer() method.
    """
    batch = {
        'X': torch.tensor(pocket_coords, dtype=torch.float32).to(device),
        'A': torch.tensor(pocket_atom_types, dtype=torch.long).to(device), # Atom types
        'B': torch.tensor(pocket_block_types, dtype=torch.long).to(device), # Block types
        'batch_id': torch.zeros(len(pocket_coords), dtype=torch.long).to(device)
    }
    return batch
# --- End Mock ATOMICA ---


def process_complex(line, atomica_model, device, atom_vocab_size):
    """
    Processes a single line (complex) from the .jsonl file.
    This version is corrected to handle the per-block data structure.
    """
    complex_data = json.loads(line)
    
    # 1. Access the nested "data" object
    try:
        data = complex_data['data']
        x = np.array(data['X'])
        a = np.array(data['A'])
        b_blocks = np.array(data['B'])
        block_lengths = np.array(data['block_lengths'])
        seg_blocks = np.array(data['segment_ids'])
    except KeyError as e:
        print(f"Skipping complex {complex_data.get('id', 'Unknown')} due to missing key: {e}")
        return None

    # Verify data integrity
    if not (len(b_blocks) == len(block_lengths) == len(seg_blocks)):
        print(f"Skipping complex {complex_data.get('id', 'Unknown')}: Block array lengths mismatch.")
        return None
    if not (sum(block_lengths) == len(x) == len(a)):
        print(f"Skipping complex {complex_data.get('id', 'Unknown')}: Atom array length mismatch.")
        return None

    # 2. Reconstruct per-atom arrays from blocks
    pocket_coords_list = []
    pocket_atom_types_list = []
    pocket_block_types_list = []
    ligand_coords_list = []
    ligand_atom_types_list = []

    
    atom_count = 0
    for i in range(len(seg_blocks)):
        length = block_lengths[i]
        block_atoms_x = x[atom_count : atom_count + length]
        block_atoms_a = a[atom_count : atom_count + length]
        block_type_b = b_blocks[i]
        
        if seg_blocks[i] == 0:  # This is a Pocket block
            pocket_coords_list.append(block_atoms_x)
            pocket_atom_types_list.append(block_atoms_a)
            # Create the per-atom block type array for ATOMICA
            pocket_block_types_list.append(np.full(length, block_type_b))
        
        elif seg_blocks[i] == 1:  # This is a Ligand block
            ligand_coords_list.append(block_atoms_x)
            ligand_atom_types_list.append(block_atoms_a)
            
        atom_count += length

    # 3. Concatenate lists into final numpy arrays
    # Check for empty pocket or ligand
    if not pocket_coords_list:
        print(f"Warning: Complex {complex_data.get('id', 'Unknown')} has no pocket atoms. Skipping.")
        return None
    if not ligand_coords_list:
        print(f"Warning: Complex {complex_data.get('id', 'Unknown')} has no ligand atoms. Skipping.")
        return None

    # Create "Raw Pocket Graph" (Context)
    pocket_coords = np.concatenate(pocket_coords_list, axis=0)
    pocket_atom_types = np.concatenate(pocket_atom_types_list, axis=0)
    pocket_block_types = np.concatenate(pocket_block_types_list, axis=0)
    
    # Create "Target Ligand Graph" (Target)
    ligand_coords = np.concatenate(ligand_coords_list, axis=0)
    ligand_atom_types = np.concatenate(ligand_atom_types_list, axis=0)

    # 4. Generate Embeddings (The "Upgrade")
    atomica_batch = format_atomica_batch(pocket_coords, pocket_atom_types, pocket_block_types, device)
    
    with torch.no_grad():
        atomica_output = atomica_model.infer(atomica_batch)
    
    pocket_atomica_embeddings = atomica_output.unit_repr.cpu().numpy()

    # 5. Save the New Data
    # One-hot encode ligand features using the provided vocab size
    try:
        ligand_one_hot = np.eye(atom_vocab_size)[ligand_atom_types]
    except IndexError as e:
        print(f"ERROR: Atom index {np.max(ligand_atom_types)} is out of bounds for vocab size {atom_vocab_size}.")
        print(f"Skipping complex {complex_data.get('id', 'Unknown')}. Please check --atom_vocab_size argument.")
        return None

    new_data = {
        'lig_coords': ligand_coords.astype(np.float32),
        'lig_one_hot': ligand_one_hot.astype(np.float32),
        'pocket_coords': pocket_coords.astype(np.float32),
        'pocket_atomica_embeddings': pocket_atomica_embeddings.astype(np.float32),
        'name': complex_data.get('id', 'complex')
    }
    
    return new_data


def main(args):
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the pre-trained ATOMICA model
    atomica_model = load_atomica_model(args.atomica_model_path, args.atomica_embed_dim).to(device).eval()

    print(f"Starting processing of {input_file}...")
    print(f"Saving processed files to {output_dir}")
    print(f"Using ATOM vocab size: {args.atom_vocab_size}")

    count = 0
    with gzip.open(input_file, 'rt', encoding='utf-8') as f:
        for line in tqdm(f):
            if not line.strip():
                continue
                
            processed_data = process_complex(line, atomica_model, device, args.atom_vocab_size)
            
            if processed_data:
                # Save as a .pt file
                file_name = f"complex_{count:06d}.pt"
                file_path = output_dir / file_name
                
                # Convert numpy arrays to tensors for saving
                tensor_data = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v 
                               for k, v in processed_data.items()}
                
                torch.save(tensor_data, file_path)
                count += 1

    print(f"\nProcessing complete.")
    print(f"Successfully processed and saved {count} complexes to {output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Pre-process PL.jsonl.gz with ATOMICA embeddings")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input PL.jsonl.gz file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the new .pt files.")
    parser.add_argument("--atomica_model_path", type=str, default="mock", 
                        help="Path to the pre-trained ATOMICA model checkpoint. (Default: 'mock')")
    parser.add_argument("--atomica_embed_dim", type=int, default=128,
                        help="Embedding dimension of the ATOMICA model. (Default: 128)")
    # NEW ARGUMENT
    parser.add_argument("--atom_vocab_size", type=int, default=19,
                        help="Total number of atom types for one-hot encoding. "
                             "Default=19 (to fit max index 18 from example).")
    
    args = parser.parse_args()
    main(args)
import pickle
import numpy as np
from typing import List, Dict, Any
from numpy.typing import NDArray

def isolate_pocket_embeddings_vectorized(atomica_output_path: str) -> List[Dict[str, NDArray[Any]]]:
    """
    Processes an ATOMICA output file to isolate protein pocket embeddings using
    vectorized NumPy operations for high performance.

    This function loads a pickle file containing embeddings for protein-ligand
    complexes. It filters these embeddings to retain only those corresponding
    to the protein pocket (segment_id = 0), discarding the ligand embeddings.

    Args:
        atomica_output_path: The file path to the pickled output from ATOMICA.
                             This file should contain a list of dictionaries,
                             each representing a complex.

    Returns:
        A list of new dictionaries, where each contains the complex 'id' and
        the filtered 'pocket_block_embeddings' and 'pocket_atom_embeddings'.
    """
    with open(atomica_output_path, 'rb') as f:
        all_complex_data = pickle.load(f)

    processed_pockets = []

    for complex_data in all_complex_data:
        
        segment_ids = np.asarray(complex_data['segment_ids'])
        block_embeddings = np.asarray(complex_data['block_embedding'])
        atom_embeddings = np.asarray(complex_data['atom_embedding'])
        block_lengths = np.asarray(complex_data['block_lengths'])

        # --- 1. Create a boolean mask for protein blocks (residues) ---
        # This mask is True where the segment ID is 0 (protein) and False otherwise.
        is_protein_block_mask = (segment_ids == 0)

        # --- 2. Isolate Block (Residue) Embeddings for the Pocket ---
        pocket_block_embeddings = block_embeddings[is_protein_block_mask]

        # --- 3. Isolate Atom Embeddings using Vectorization ---
        is_protein_atom_mask = np.repeat(is_protein_block_mask, block_lengths)
        pocket_atom_embeddings = atom_embeddings[is_protein_atom_mask]
        # --- 4. Store the Processed Data ---
        processed_pockets.append({
            'id': complex_data['id'],
            'pocket_block_embeddings': pocket_block_embeddings,
            'pocket_atom_embeddings': pocket_atom_embeddings
        })

    return processed_pockets


if __name__ == '__main__':

    # Path to the pkl file of ATOMICA outputs
    input_file = 'data/example/example_outputs_embedded.pkl'
    # Path to save the cleaned pocket embeddings
    output_file = 'data/example/example_outputs_embedded_pocket.pkl'


    print(f"Loading data from '{input_file}' and isolating pocket embeddings...")
    isolated_data = isolate_pocket_embeddings_vectorized(input_file)
    print("Processing complete.")

    with open(output_file, 'wb') as f:
        pickle.dump(isolated_data, f)
    print(f"Cleaned pocket embeddings saved to '{output_file}'.")

    # before and after
    print("\nVerification of the first processed pocket:")

    # Load the "before" shapes
    with open(input_file, 'rb') as f:
        original_data = pickle.load(f)
    first_original_complex = original_data[0]
    
    print(f"  Complex ID: {first_original_complex['id']}")
    
    print("\n--- Before Isolation (Full Complex) ---")
    print(f"  Shape of original block embeddings: {np.asarray(first_original_complex['block_embedding']).shape}")
    print(f"  Shape of original atom embeddings:  {np.asarray(first_original_complex['atom_embedding']).shape}")

    print("\n--- After Isolation (Pocket Only) ---")
    print(f"  Shape of pocket block embeddings:   {isolated_data[0]['pocket_block_embeddings'].shape}")
    print(f"  Shape of pocket atom embeddings:    {isolated_data[0]['pocket_atom_embeddings'].shape}")

    
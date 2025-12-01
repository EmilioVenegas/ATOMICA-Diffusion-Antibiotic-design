#!/usr/bin/env python3
"""
Simple test script for RL_loop.py

Tests the pipeline with example ligands from DiffSBDD
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from RL_loop import RLLoop


def test_example_ligands():
    """Test with DiffSBDD example ligands"""
    print("="*60)
    print("Testing RL Loop with DiffSBDD Example Ligands")
    print("="*60)
    
    # Path to DiffSBDD examples
    example_dir = Path(__file__).parent.parent / "DiffSBDD" / "example"
    
    if not example_dir.exists():
        print(f"Error: Example directory not found: {example_dir}")
        return False
    
    # Initialize RL loop
    rl = RLLoop(verbose=True)
    
    # Test loading single SDF
    print("\n--- Test 1: Load single SDF file ---")
    sdf_file = example_dir / "3rfm_B_CFF.sdf"
    if sdf_file.exists():
        molecules = rl.load_molecules_from_sdf(sdf_file)
        print(f"Loaded {len(molecules)} molecules")
        
        # Convert to SMILES
        smiles = rl.convert_to_smiles(molecules)
        print(f"SMILES: {smiles}")
    else:
        print(f"Warning: File not found: {sdf_file}")
    
    # Test loading directory
    print("\n--- Test 2: Load directory of SDFs ---")
    mol_dict = rl.load_molecules_from_directory(example_dir)
    total = sum(len(mols) for mols in mol_dict.values())
    print(f"Loaded {total} molecules from {len(mol_dict)} files")
    
    # Test ADMET scoring (if available)
    if rl.admet_model:
        print("\n--- Test 3: ADMET-AI Scoring ---")
        all_smiles = []
        for mols in mol_dict.values():
            all_smiles.extend(rl.convert_to_smiles(mols))
        
        valid_smiles = [s for s in all_smiles if s is not None]
        if valid_smiles:
            scores = rl.score_with_admet(valid_smiles[:5])  # Test with first 5
            print(f"Score shape: {scores.shape}")
            print(f"Columns: {list(scores.columns)}")
    else:
        print("\n--- Test 3: ADMET-AI Scoring ---")
        print("Skipped (ADMET-AI not available)")
    
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)
    
    return True


def test_full_pipeline():
    """Test the complete pipeline"""
    print("\n")
    print("="*60)
    print("Testing Full Pipeline")
    print("="*60)
    
    example_dir = Path(__file__).parent.parent / "DiffSBDD" / "example"
    output_file = Path(__file__).parent / "results" / "test_output.csv"
    
    if not example_dir.exists():
        print(f"Error: Example directory not found: {example_dir}")
        return False
    
    rl = RLLoop(verbose=True)
    rl.run_pipeline(
        input_path=example_dir,
        output_path=output_file,
        top_k=None
    )
    
    # Check output
    if output_file.exists():
        print(f"\n✓ Output file created: {output_file}")
        import pandas as pd
        df = pd.read_csv(output_file)
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        print(f"\nTop 3 molecules:")
        print(df.head(3).to_string())
    else:
        print(f"\n✗ Output file not created: {output_file}")
    
    return True


if __name__ == "__main__":
    # Run tests
    test_example_ligands()
    test_full_pipeline()


#!/usr/bin/env python3
"""
RL Loop for ADMET-AI Guided Ligand Optimization

This script implements the initial RL feedback loop:
1. Load ligands from SDF files
2. Convert to SMILES
3. Score with ADMET-AI
4. Rank by predicted properties
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# Try to import ADMET-AI
try:
    from admet_ai import ADMETModel
    ADMET_AVAILABLE = True
except ImportError:
    ADMET_AVAILABLE = False
    print("Warning: ADMET-AI not available. Install with: pip install admet-ai")


class RLLoop:
    """Reinforcement Learning Loop for Ligand Optimization"""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize RL Loop
        
        Args:
            verbose: Print progress messages
        """
        self.verbose = verbose
        self.admet_model = None
        
        if ADMET_AVAILABLE:
            if self.verbose:
                print("Loading ADMET-AI model...")
            try:
                self.admet_model = ADMETModel()
                if self.verbose:
                    print("✓ ADMET-AI model loaded successfully")
            except Exception as e:
                print(f"Warning: Failed to load ADMET-AI model: {e}")
    
    def load_molecules_from_sdf(self, sdf_path: Path) -> List[Chem.Mol]:
        """
        Load molecules from SDF file
        
        Args:
            sdf_path: Path to SDF file
            
        Returns:
            List of RDKit molecule objects
        """
        if self.verbose:
            print(f"Loading molecules from {sdf_path}...")
        
        molecules = []
        supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False)
        
        for idx, mol in enumerate(supplier):
            if mol is not None:
                try:
                    # Try to sanitize
                    Chem.SanitizeMol(mol)
                    molecules.append(mol)
                except Exception as e:
                    if self.verbose:
                        print(f"  Warning: Molecule {idx} failed sanitization: {e}")
            else:
                if self.verbose:
                    print(f"  Warning: Molecule {idx} could not be loaded")
        
        if self.verbose:
            print(f"✓ Loaded {len(molecules)} valid molecules")
        
        return molecules
    
    def load_molecules_from_directory(self, dir_path: Path) -> Dict[str, List[Chem.Mol]]:
        """
        Load all SDF files from a directory
        
        Args:
            dir_path: Path to directory containing SDF files
            
        Returns:
            Dictionary mapping filename to list of molecules
        """
        if self.verbose:
            print(f"Scanning directory: {dir_path}")
        
        all_molecules = {}
        sdf_files = list(Path(dir_path).glob("*.sdf"))
        
        if not sdf_files:
            print(f"Warning: No SDF files found in {dir_path}")
            return all_molecules
        
        for sdf_file in sdf_files:
            molecules = self.load_molecules_from_sdf(sdf_file)
            if molecules:
                all_molecules[sdf_file.name] = molecules
        
        total = sum(len(mols) for mols in all_molecules.values())
        if self.verbose:
            print(f"✓ Loaded {total} molecules from {len(all_molecules)} files")
        
        return all_molecules
    
    def convert_to_smiles(self, molecules: List[Chem.Mol], 
                         remove_stereo: bool = True) -> List[str]:
        """
        Convert RDKit molecules to SMILES strings
        
        Args:
            molecules: List of RDKit molecules
            remove_stereo: Remove stereochemistry information
            
        Returns:
            List of SMILES strings
        """
        if self.verbose:
            print("Converting molecules to SMILES...")
        
        smiles_list = []
        for mol in molecules:
            try:
                # Create a copy to avoid modifying original
                mol_copy = Chem.Mol(mol)
                
                if remove_stereo:
                    Chem.RemoveStereochemistry(mol_copy)
                
                # Remove explicit hydrogens
                mol_copy = Chem.RemoveHs(mol_copy)
                
                # Convert to SMILES
                smiles = Chem.MolToSmiles(mol_copy)
                smiles_list.append(smiles)
                
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Failed to convert molecule to SMILES: {e}")
                smiles_list.append(None)
        
        valid_count = sum(1 for s in smiles_list if s is not None)
        if self.verbose:
            print(f"✓ Converted {valid_count}/{len(molecules)} molecules to SMILES")
        
        return smiles_list
    
    def score_with_admet(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        Score molecules using ADMET-AI
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            DataFrame with ADMET predictions
        """
        if not self.admet_model:
            print("Error: ADMET-AI model not available")
            return pd.DataFrame({'smiles': smiles_list})
        
        # Filter out None values
        valid_smiles = [s for s in smiles_list if s is not None]
        
        if not valid_smiles:
            print("Error: No valid SMILES to score")
            return pd.DataFrame()
        
        if self.verbose:
            print(f"Scoring {len(valid_smiles)} molecules with ADMET-AI...")
        
        try:
            # Run ADMET-AI predictions
            predictions = self.admet_model.predict(smiles=valid_smiles)
            
            if self.verbose:
                print(f"✓ ADMET-AI predictions complete")
                print(f"  Properties predicted: {list(predictions.columns)}")
            
            return predictions
            
        except Exception as e:
            print(f"Error during ADMET-AI prediction: {e}")
            return pd.DataFrame({'smiles': valid_smiles})
    
    def compute_composite_score(self, df: pd.DataFrame, 
                                properties: Optional[List[str]] = None) -> pd.Series:
        """
        Compute composite score from ADMET properties
        
        Args:
            df: DataFrame with ADMET predictions
            properties: List of property names to use (uses all if None)
            
        Returns:
            Series with composite scores
        """
        if properties is None:
            # Use all numeric columns except 'smiles'
            properties = [col for col in df.columns 
                         if col != 'smiles' and pd.api.types.is_numeric_dtype(df[col])]
        
        if not properties:
            print("Warning: No numeric properties found for scoring")
            return pd.Series([0.0] * len(df))
        
        # Simple average (could be weighted in future)
        # Normalize each property to [0, 1] range
        # Reset index to avoid duplicate label issues
        df_reset = df.reset_index(drop=True)
        scores = pd.Series([0.0] * len(df_reset), index=df_reset.index)

        for prop in properties:
            if prop in df_reset.columns:
                values = df_reset[prop]
                # Skip if all NaN
                if values.notna().any():
                    # Normalize to [0, 1]
                    min_val = values.min()
                    max_val = values.max()
                    if max_val > min_val:
                        normalized = (values - min_val) / (max_val - min_val)
                    else:
                        normalized = pd.Series([0.5] * len(values), index=df_reset.index)
                    scores = scores + normalized.fillna(0.5)  # Use + instead of +=
        
        # Average across properties
        if len(properties) > 0:
            scores = scores / len(properties)
        
        return scores
    
    def rank_by_score(self, df: pd.DataFrame, 
                     score_col: str = 'composite_score',
                     ascending: bool = False) -> pd.DataFrame:
        """
        Rank molecules by score
        
        Args:
            df: DataFrame with scores
            score_col: Column name to rank by
            ascending: Rank in ascending order (True) or descending (False)
            
        Returns:
            DataFrame sorted by rank with 'rank' column added
        """
        if score_col not in df.columns:
            print(f"Warning: Score column '{score_col}' not found")
            df['rank'] = range(1, len(df) + 1)
            return df
        
        # Sort by score
        df_sorted = df.sort_values(score_col, ascending=ascending).reset_index(drop=True)
        
        # Add rank column
        df_sorted.insert(0, 'rank', range(1, len(df_sorted) + 1))
        
        if self.verbose:
            print(f"✓ Ranked {len(df_sorted)} molecules by {score_col}")
        
        return df_sorted
    
    def save_results(self, df: pd.DataFrame, output_path: Path, top_k: Optional[int] = None):
        """
        Save results to CSV
        
        Args:
            df: DataFrame with results
            output_path: Output file path
            top_k: Only save top k molecules (saves all if None)
        """
        if top_k is not None:
            df = df.head(top_k)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        
        if self.verbose:
            print(f"✓ Saved {len(df)} results to {output_path}")
    
    def run_pipeline(self, input_path: Path, output_path: Path, 
                top_k: Optional[int] = None,
                save_top_sdf: bool = False,
                top_sdf_count: int = 1):
    """
    Run the complete RL loop pipeline
    
    Args:
        input_path: Path to SDF file or directory
        output_path: Path for output CSV
        top_k: Only save top k molecules in CSV
        save_top_sdf: Also save top molecule(s) as SDF
        top_sdf_count: Number of top molecules to save as SDF (default: 1)
    """
    print("="*60)
    print("RL LOOP: ADMET-AI Guided Ligand Optimization")
    print("="*60)
    
    # Load molecules
    input_path = Path(input_path)
    if input_path.is_file():
        molecules = self.load_molecules_from_sdf(input_path)
        mol_dict = {input_path.name: molecules}
    else:
        mol_dict = self.load_molecules_from_directory(input_path)
    
    # ... existing code for creating df and scoring ...
    
    # Save results
    self.save_results(df, output_path, top_k=top_k)
    
    # Save top molecule(s) as SDF if requested
    if save_top_sdf and not df.empty:
        # Create molecule_id to mol mapping
        mol_id_to_mol = {}
        for file_mols in mol_dict.values():
            for i, mol in enumerate(file_mols):
                mol_id = f"{input_path.stem}_{i}"
                mol_id_to_mol[mol_id] = mol
        
        # Determine output path for SDF
        sdf_output = output_path.parent / f"{output_path.stem}_top{top_sdf_count}.sdf"
        
        self.save_top_molecules_sdf(df, mol_id_to_mol, sdf_output, top_k=top_sdf_count)
    
    print("="*60)
    print("Pipeline complete!")
    if not df.empty and 'composite_score' in df.columns:
        print(f"Top molecule: {df.iloc[0]['molecule_id']} (score: {df.iloc[0]['composite_score']:.3f})")
    print("="*60)
    
    def save_top_molecules_sdf(self, df: pd.DataFrame, molecules_dict: dict, 
                           output_path: Path, top_k: int = 1):
    """
    Save top-k molecules as SDF file
    
    Args:
        df: DataFrame with ranked results (must have 'smiles' column)
        molecules_dict: Dictionary mapping molecule_id to RDKit mol objects
        output_path: Output SDF file path
        top_k: Number of top molecules to save (default: 1)
    """
    if df.empty:
        if self.verbose:
            print("Warning: No molecules to save")
        return
    
    # Get top-k molecules
    top_df = df.head(top_k)
    
    # Create mapping from SMILES to mol
    smiles_to_mol = {}
    for mol_id, mol in molecules_dict.items():
        if mol is not None:
            smiles = Chem.MolToSmiles(mol)
            smiles_to_mol[smiles] = mol
    
    # Write SDF
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    writer = Chem.SDWriter(str(output_path))
    saved_count = 0
    
    for idx, row in top_df.iterrows():
        smiles = row['smiles']
        if smiles in smiles_to_mol:
            mol = smiles_to_mol[smiles]
            # Add properties to molecule
            mol.SetProp('_Name', str(row.get('molecule_id', f'mol_{idx}')))
            mol.SetProp('rank', str(row.get('rank', idx + 1)))
            mol.SetProp('composite_score', str(row.get('composite_score', 0.0)))
            mol.SetProp('smiles', smiles)
            
            writer.write(mol)
            saved_count += 1
        else:
            if self.verbose:
                print(f"  Warning: Could not find molecule for SMILES: {smiles[:50]}...")
    
    writer.close()
    
    if self.verbose:
        print(f"✓ Saved top-{top_k} molecule(s) to {output_path}")
    
    return saved_count


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='RL Loop: Score and rank ligands using ADMET-AI',
        # ... existing args ...
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='Input SDF file or directory')
    parser.add_argument('-o', '--output', required=True,
                       help='Output CSV file')
    parser.add_argument('-k', '--top_k', type=int, default=None,
                       help='Only save top k molecules (default: all)')
    parser.add_argument('--save_top_sdf', action='store_true',
                       help='Also save top molecule(s) as SDF file')
    parser.add_argument('--top_sdf_count', type=int, default=1,
                       help='Number of top molecules to save as SDF (default: 1)')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress progress messages')
    
    args = parser.parse_args()
    
    # Run pipeline
    rl_loop = RLLoop(verbose=not args.quiet)
    rl_loop.run_pipeline(
        input_path=args.input,
        output_path=args.output,
        top_k=args.top_k,
        save_top_sdf=args.save_top_sdf,
        top_sdf_count=args.top_sdf_count
    )


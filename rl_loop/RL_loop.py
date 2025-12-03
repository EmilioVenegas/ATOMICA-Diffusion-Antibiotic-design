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
        if self.verbose:
            print(f"Loading molecules from {sdf_path}...")
        
        molecules = []
        supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False)
        
        for idx, mol in enumerate(supplier):
            if mol is not None:
                try:
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
    
    def convert_to_smiles(self, molecules: List[Chem.Mol], remove_stereo: bool = True) -> List[str]:
        if self.verbose:
            print("Converting molecules to SMILES...")
        
        smiles_list = []
        for mol in molecules:
            try:
                mol_copy = Chem.Mol(mol)
                
                if remove_stereo:
                    Chem.RemoveStereochemistry(mol_copy)
                
                mol_copy = Chem.RemoveHs(mol_copy)
                
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
        if not self.admet_model:
            print("Error: ADMET-AI model not available")
            return pd.DataFrame({'smiles': smiles_list})
        
        valid_smiles = [s for s in smiles_list if s is not None]
        
        if not valid_smiles:
            print("Error: No valid SMILES to score")
            return pd.DataFrame()
        
        if self.verbose:
            print(f"Scoring {len(valid_smiles)} molecules with ADMET-AI...")
        
        try:
            predictions = self.admet_model.predict(smiles=valid_smiles)
            
            if self.verbose:
                print(f"✓ ADMET-AI predictions complete")
                print(f"  Properties predicted: {list(predictions.columns)}")
            
            return predictions
            
        except Exception as e:
            print(f"Error during ADMET-AI prediction: {e}")
            return pd.DataFrame({'smiles': valid_smiles})
    
    def compute_composite_score(self, df: pd.DataFrame, properties: Optional[List[str]] = None) -> pd.Series:
        if properties is None:
            properties = [col for col in df.columns 
                          if col != 'smiles' and pd.api.types.is_numeric_dtype(df[col])]
        
        if not properties:
            print("Warning: No numeric properties found for scoring")
            return pd.Series([0.0] * len(df))
        
        df_reset = df.reset_index(drop=True)
        scores = pd.Series([0.0] * len(df_reset), index=df_reset.index)

        for prop in properties:
            if prop in df_reset.columns:
                values = df_reset[prop]
                if values.notna().any():
                    min_val = values.min()
                    max_val = values.max()
                    if max_val > min_val:
                        normalized = (values - min_val) / (max_val - min_val)
                    else:
                        normalized = pd.Series([0.5] * len(values), index=df_reset.index)
                    scores = scores + normalized.fillna(0.5)
        
        if len(properties) > 0:
            scores = scores / len(properties)
        
        return scores
    
    def rank_by_score(self, df: pd.DataFrame, score_col: str = 'composite_score', ascending: bool = False) -> pd.DataFrame:
        if score_col not in df.columns:
            print(f"Warning: Score column '{score_col}' not found")
            df['rank'] = range(1, len(df) + 1)
            return df
        
        df_sorted = df.sort_values(score_col, ascending=ascending).reset_index(drop=True)
        df_sorted.insert(0, 'rank', range(1, len(df_sorted) + 1))
        
        if self.verbose:
            print(f"✓ Ranked {len(df_sorted)} molecules by {score_col}")
        
        return df_sorted
    
    def save_results(self, df: pd.DataFrame, output_path: Path, top_k: Optional[int] = None):
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
        """
        print("=" * 60)
        print("RL LOOP: ADMET-AI Guided Ligand Optimization")
        print("=" * 60)

        # Load molecules (single SDF or directory)
        input_path = Path(input_path)
        if input_path.is_file():
            molecules = self.load_molecules_from_sdf(input_path)
            mol_dict = {input_path.name: molecules}
        else:
            mol_dict = self.load_molecules_from_directory(input_path)

        # Build dataframe of molecules and SMILES
        if not mol_dict:
            print("Error: No molecules loaded")
            return

        all_data = []
        for filename, molecules in mol_dict.items():
            smiles_list = self.convert_to_smiles(molecules)
            for idx, (mol, smiles) in enumerate(zip(molecules, smiles_list)):
                if smiles is not None:
                    all_data.append({
                        "molecule_id": f"{filename}_{idx}",
                        "smiles": smiles,
                        "source_file": filename,
                    })

        df = pd.DataFrame(all_data)

        if df.empty:
            print("Error: No valid SMILES generated")
            return

        # Score with ADMET-AI
        admet_scores = self.score_with_admet(df["smiles"].tolist())

        if not admet_scores.empty:
            admet_scores_reset = admet_scores.reset_index()
            if "smiles" in admet_scores_reset.columns:
                df = df.merge(admet_scores_reset, on="smiles", how="left")
            else:
                admet_scores_reset = admet_scores_reset.rename(
                    columns={admet_scores_reset.columns[0]: "smiles"}
                )
                df = df.merge(admet_scores_reset, on="smiles", how="left")

            # Compute composite score and rank
            df["composite_score"] = self.compute_composite_score(df)
            df = self.rank_by_score(df, score_col="composite_score", ascending=False)

        # Save results CSV
        self.save_results(df, output_path, top_k=top_k)

        # Save top molecules as SDF if requested
        if save_top_sdf and not df.empty:
            mol_id_to_mol = {}
            for filename, molecules in mol_dict.items():
                for i, mol in enumerate(molecules):
                    mol_id = f"{filename}_{i}"
                    mol_id_to_mol[mol_id] = mol

            sdf_output = output_path.parent / f"{output_path.stem}_top{top_sdf_count}.sdf"
            self.save_top_molecules_sdf(
                df, mol_id_to_mol, sdf_output, top_k=top_sdf_count
            )

        print("=" * 60)
        print("Pipeline complete!")
        if not df.empty and "composite_score" in df.columns:
            print(
                f"Top molecule: {df.iloc[0]['molecule_id']} "
                f"(score: {df.iloc[0]['composite_score']:.3f})"
            )
        print("=" * 60)
    
    def save_top_molecules_sdf(self, df: pd.DataFrame, molecules_dict: dict, 
                               output_path: Path, top_k: int = 1):
        """
        Save top-k molecules as SDF file
        """
        if df.empty:
            if self.verbose:
                print("Warning: No molecules to save")
            return
        
        top_df = df.head(top_k)
        
        # SMILES → molecule lookup
        smiles_to_mol = {}
        for mol_id, mol in molecules_dict.items():
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                smiles_to_mol[smiles] = mol
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        writer = Chem.SDWriter(str(output_path))
        saved_count = 0
        
        for idx, row in top_df.iterrows():
            smiles = row['smiles']
            mol = smiles_to_mol.get(smiles, None)
            if mol is None:
                if self.verbose:
                    print(f"  Warning: Could not find molecule for SMILES: {smiles[:50]}...")
                continue
            
            mol.SetProp('_Name', str(row.get('molecule_id', f'mol_{idx}')))
            mol.SetProp('rank', str(row.get('rank', idx + 1)))
            mol.SetProp('composite_score', str(row.get('composite_score', 0.0)))
            mol.SetProp('smiles', smiles)
            
            writer.write(mol)
            saved_count += 1
        
        writer.close()
        
        if self.verbose:
            print(f"✓ Saved top-{top_k} molecule(s) to {output_path}")
        
        return saved_count


def main():
    parser = argparse.ArgumentParser(
        description='RL Loop: Score and rank ligands using ADMET-AI'
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
                        help='Number of top molecules to save as SDF')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress progress messages')
    
    args = parser.parse_args()
    
    rl_loop = RLLoop(verbose=not args.quiet)
    rl_loop.run_pipeline(
        input_path=args.input,
        output_path=args.output,
        top_k=args.top_k,
        save_top_sdf=args.save_top_sdf,
        top_sdf_count=args.top_sdf_count
    )
    
if __name__ == "__main__":
    main()

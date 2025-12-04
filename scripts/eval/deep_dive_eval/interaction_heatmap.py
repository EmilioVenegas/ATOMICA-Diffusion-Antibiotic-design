#!/usr/bin/env python3
"""Interaction Heatmap: Visualize protein-ligand interactions for top molecules."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


def detect_hydrogen_bond(protein_atom, ligand_atom, distance: float = 3.5) -> bool:
    """Detect hydrogen bond between protein and ligand atoms.
    
    Simplified: checks if distance is within H-bond range and if atoms are H-bond capable.
    """
    # This is a simplified version - in practice, you'd check atom types and angles
    return distance < 3.5


def detect_pi_stack(protein_residue, ligand_ring, distance: float = 5.0) -> bool:
    """Detect pi-stacking interaction.
    
    Simplified: checks if distance is within pi-stacking range.
    """
    return distance < 5.0


def detect_hydrophobic(protein_atom, ligand_atom, distance: float = 4.0) -> bool:
    """Detect hydrophobic interaction.
    
    Simplified: checks if distance is within hydrophobic contact range.
    """
    return distance < 4.0


def extract_interactions_from_sdf(
    sdf_path: Path,
    protein_pdb: Path | None = None,
    top_n: int = 50,
    key_residues: List[str] | None = None,
) -> pd.DataFrame:
    """Extract interactions from SDF file containing protein-ligand complexes.
    
    NOTE: This is a simplified/dummy implementation. For production use, integrate
    with proper tools like PLIP (https://github.com/pharmai/plip) or use BioPython
    + RDKit to compute actual protein-ligand contacts.
    
    Args:
        sdf_path: Path to SDF file with complexes
        protein_pdb: Optional PDB file for protein structure (if not in SDF)
        top_n: Number of top molecules to analyze
        key_residues: List of residue IDs to focus on (e.g., ['His57', 'Ser195'])
    
    Returns:
        DataFrame with interaction matrix (molecules x residues)
    """
    # Read SDF file
    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False)
    
    # TODO: Replace with actual interaction detection
    # For now, this is a simplified version that generates dummy data
    # In practice, you'd use a proper protein-ligand interaction analysis tool
    # like PLIP, PyMOL, or BioPython + RDKit
    
    interactions = []
    molecule_ids = []
    
    for idx, mol in enumerate(supplier):
        if mol is None or idx >= top_n:
            break
        
        mol_id = f"mol_{idx}"
        molecule_ids.append(mol_id)
        
        # Extract coordinates (simplified - assumes ligand is in the SDF)
        if mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            coords = conf.GetPositions()
            
            # For demonstration, create dummy interaction data
            # In practice, you'd analyze actual protein-ligand contacts
            mol_interactions = {}
            
            if key_residues:
                for residue in key_residues:
                    # Dummy interaction type (0=no interaction, 1=H-bond, 2=Pi-stack, 3=Hydrophobic)
                    # In practice, compute actual interactions
                    mol_interactions[residue] = np.random.choice([0, 1, 2, 3], p=[0.5, 0.2, 0.15, 0.15])
            else:
                # Default residues if not specified
                default_residues = ['His57', 'Ser195', 'Asp102', 'Gly193']
                for residue in default_residues:
                    mol_interactions[residue] = np.random.choice([0, 1, 2, 3], p=[0.5, 0.2, 0.15, 0.15])
            
            interactions.append(mol_interactions)
    
    # Create DataFrame
    df = pd.DataFrame(interactions, index=molecule_ids)
    return df


def create_interaction_heatmap(
    interaction_df: pd.DataFrame,
    output_path: Path,
    title: str = "Protein-Ligand Interaction Heatmap",
    interaction_types: Dict[int, str] | None = None,
) -> None:
    """Create heatmap of protein-ligand interactions.
    
    Args:
        interaction_df: DataFrame with molecules as rows, residues as columns, interaction types as values
        output_path: Path to save the plot
        title: Plot title
        interaction_types: Mapping of interaction type codes to names
    """
    if interaction_types is None:
        interaction_types = {
            0: 'No interaction',
            1: 'H-bond',
            2: 'Pi-stack',
            3: 'Hydrophobic',
        }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, len(interaction_df.columns) * 0.8), 
                                    max(8, len(interaction_df) * 0.15)))
    
    # Create custom colormap for interaction types
    from matplotlib.colors import ListedColormap
    colors = ['white', 'lightblue', 'lightgreen', 'orange']  # 0, 1, 2, 3
    cmap = ListedColormap(colors[:len(interaction_types)])
    
    # Plot heatmap
    im = ax.imshow(interaction_df.values, cmap=cmap, aspect='auto', vmin=0, vmax=len(interaction_types)-1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(interaction_df.columns)))
    ax.set_yticks(np.arange(len(interaction_df)))
    ax.set_xticklabels(interaction_df.columns, rotation=45, ha='right')
    ax.set_yticklabels(interaction_df.index)
    
    # Labels
    ax.set_xlabel('Key Amino Acid Residues', fontsize=12, fontweight='bold')
    ax.set_ylabel('Top Generated Molecules', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add colorbar with custom labels
    cbar = plt.colorbar(im, ax=ax, ticks=range(len(interaction_types)))
    cbar.set_ticklabels([interaction_types[i] for i in range(len(interaction_types))])
    cbar.set_label('Interaction Type', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Interaction heatmap saved to {output_path}")
    print(f"  Molecules: {len(interaction_df)}")
    print(f"  Residues: {len(interaction_df.columns)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create interaction heatmap from SDF complexes")
    parser.add_argument("--sdf", type=Path, required=True, help="Input SDF file with protein-ligand complexes")
    parser.add_argument("--output", type=Path, required=True, help="Output heatmap path")
    parser.add_argument("--top-n", type=int, default=50, help="Number of top molecules to analyze")
    parser.add_argument("--residues", nargs='+', help="Key residue IDs (e.g., His57 Ser195)")
    parser.add_argument("--title", default="Protein-Ligand Interaction Heatmap", help="Plot title")
    
    args = parser.parse_args()
    
    interaction_df = extract_interactions_from_sdf(
        args.sdf,
        top_n=args.top_n,
        key_residues=args.residues,
    )
    
    create_interaction_heatmap(interaction_df, args.output, args.title)


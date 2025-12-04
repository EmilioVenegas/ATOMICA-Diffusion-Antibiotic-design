#!/usr/bin/env python3
"""Judge 2: The Physicist - AutoDock Vina scoring for protein-ligand docking."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict

import pandas as pd
from rdkit import Chem


def check_vina_installed() -> bool:
    """Check if Vina is installed and available."""
    try:
        result = subprocess.run(['vina', '--help'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def prepare_receptor_pdbqt(pdb_path: Path, output_pdbqt: Path | None = None) -> Path:
    """Prepare receptor PDB file for Vina (convert to PDBQT).
    
    Args:
        pdb_path: Path to receptor PDB file
        output_pdbqt: Optional output path (default: same name with .pdbqt extension)
    
    Returns:
        Path to prepared PDBQT file
    """
    if output_pdbqt is None:
        output_pdbqt = pdb_path.with_suffix('.pdbqt')
    
    # Use prepare_receptor4.py from AutoDockTools
    cmd = ['prepare_receptor4.py', '-r', str(pdb_path), '-O', str(output_pdbqt)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to prepare receptor: {result.stderr}")
    
    return output_pdbqt


def prepare_ligand_pdbqt(sdf_path: Path, mol_idx: int = 0, output_pdbqt: Path | None = None) -> Path:
    """Prepare ligand from SDF for Vina (convert to PDBQT).
    
    Args:
        sdf_path: Path to SDF file
        mol_idx: Index of molecule in SDF (default: 0)
        output_pdbqt: Optional output path
    
    Returns:
        Path to prepared PDBQT file
    """
    if output_pdbqt is None:
        output_pdbqt = sdf_path.with_suffix('.pdbqt')
    
    # Use obabel to convert SDF to PDBQT
    cmd = ['obabel', str(sdf_path), '-f', str(mol_idx + 1), '-l', str(mol_idx + 1), '-O', str(output_pdbqt)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to prepare ligand: {result.stderr}")
    
    return output_pdbqt


def calculate_vina_score(
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    center: tuple[float, float, float] | None = None,
    size: tuple[float, float, float] = (20.0, 20.0, 20.0),
    exhaustiveness: int = 8,
    num_modes: int = 1,
) -> float:
    """Calculate Vina docking score.
    
    Args:
        receptor_pdbqt: Path to receptor PDBQT file
        ligand_pdbqt: Path to ligand PDBQT file
        center: Center of search space (x, y, z). If None, will be calculated from ligand
        size: Size of search space (x, y, z) in Angstroms
        exhaustiveness: Exhaustiveness of search (higher = more thorough, slower)
        num_modes: Number of binding modes to return
    
    Returns:
        Best Vina score (lower is better, typically negative)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        output_log = Path(tmpdir) / "vina_output.txt"
        
        # If center not provided, calculate from ligand
        if center is None:
            mol = Chem.SDMolSupplier(str(ligand_pdbqt.with_suffix('.sdf'))) if ligand_pdbqt.suffix == '.pdbqt' else None
            if mol is None:
                # Try to read from PDBQT and estimate center
                # For simplicity, use a default center
                center = (0.0, 0.0, 0.0)
            else:
                conf = mol[0].GetConformer()
                coords = conf.GetPositions()
                center = tuple(coords.mean(axis=0))
        
        cmd = [
            'vina',
            '--receptor', str(receptor_pdbqt),
            '--ligand', str(ligand_pdbqt),
            '--center_x', str(center[0]),
            '--center_y', str(center[1]),
            '--center_z', str(center[2]),
            '--size_x', str(size[0]),
            '--size_y', str(size[1]),
            '--size_z', str(size[2]),
            '--exhaustiveness', str(exhaustiveness),
            '--num_modes', str(num_modes),
            '--out', str(Path(tmpdir) / "output.pdbqt")  # Dummy output
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Vina docking failed: {result.stderr}")
        
        # Parse Vina output to extract best score
        output_lines = result.stdout.split('\n')
        for line in output_lines:
            if '   1    ' in line or line.strip().startswith('1'):
                # Format: "   1    -6.5    0.000    0.000"
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        return float(parts[1])
                    except ValueError:
                        continue
        
        # Fallback: try to find "Affinity:" line
        match = re.search(r'Affinity:\s+([+-]?[\d.]+)', result.stdout)
        if match:
            return float(match.group(1))
        
        raise RuntimeError(f"Could not parse Vina score from output: {result.stdout}")


def score_ligands_from_sdf(
    sdf_path: Path,
    receptor_pdb: Path,
    output_csv: Path | None = None,
    center: tuple[float, float, float] | None = None,
    size: tuple[float, float, float] = (20.0, 20.0, 20.0),
    exhaustiveness: int = 8,
) -> pd.DataFrame:
    """Score multiple ligands from an SDF file using Vina.
    
    Args:
        sdf_path: Path to SDF file with multiple ligands
        receptor_pdb: Path to receptor PDB file
        output_csv: Optional path to save results CSV
        center: Center of search space (x, y, z)
        size: Size of search space
        exhaustiveness: Exhaustiveness of search
    
    Returns:
        DataFrame with 'molecule_id' and 'vina_score' columns
    """
    if not check_vina_installed():
        raise RuntimeError("Vina is not installed. Please install AutoDock Vina.")
    
    # Prepare receptor
    receptor_pdbqt = prepare_receptor_pdbqt(receptor_pdb)
    
    # Read all molecules from SDF
    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False)
    results = []
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, mol in enumerate(supplier):
            if mol is None:
                continue
            
            mol_id = f"mol_{idx}"
            ligand_pdbqt = Path(tmpdir) / f"{mol_id}.pdbqt"
            
            # Save molecule to temporary SDF first
            temp_sdf = Path(tmpdir) / f"{mol_id}.sdf"
            writer = Chem.SDWriter(str(temp_sdf))
            writer.write(mol)
            writer.close()
            
            # Convert to PDBQT
            prepare_ligand_pdbqt(temp_sdf, 0, ligand_pdbqt)
            
            # Calculate Vina score
            try:
                score = calculate_vina_score(
                    receptor_pdbqt,
                    ligand_pdbqt,
                    center=center,
                    size=size,
                    exhaustiveness=exhaustiveness,
                )
                results.append({'molecule_id': mol_id, 'vina_score': score})
            except Exception as e:
                print(f"Warning: Failed to score {mol_id}: {e}")
                results.append({'molecule_id': mol_id, 'vina_score': None})
    
    df = pd.DataFrame(results)
    
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Saved Vina scores to {output_csv}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Judge 2: Vina scoring for protein-ligand docking")
    parser.add_argument("--sdf", type=Path, required=True, help="Input SDF file with ligands")
    parser.add_argument("--receptor", type=Path, required=True, help="Receptor PDB file")
    parser.add_argument("--output", type=Path, help="Output CSV file with Vina scores")
    parser.add_argument("--center", nargs=3, type=float, help="Search space center (x y z)")
    parser.add_argument("--size", nargs=3, type=float, default=[20.0, 20.0, 20.0], help="Search space size (default: 20 20 20)")
    parser.add_argument("--exhaustiveness", type=int, default=8, help="Exhaustiveness (default: 8)")
    
    args = parser.parse_args()
    
    center = tuple(args.center) if args.center else None
    size = tuple(args.size)
    
    df = score_ligands_from_sdf(
        args.sdf,
        args.receptor,
        args.output,
        center=center,
        size=size,
        exhaustiveness=args.exhaustiveness,
    )
    
    print(f"\nVina Scoring Results:")
    print(f"  Total molecules scored: {len(df)}")
    print(f"  Mean Vina score: {df['vina_score'].mean():.2f} kcal/mol")
    print(f"  Best (lowest) score: {df['vina_score'].min():.2f} kcal/mol")


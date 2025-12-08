

# Extract the protein sequence
# Get the csv file 
# Create the yamls for each ligand in the sdf file [(protein, ligand) for ligand in sdf file]

from __future__ import annotations

from pathlib import Path
from typing import List
import pandas as pd
import yaml
import re

from get_protein_sequence import get_protein_sequence_from_pdb


def _sanitize_name(name: str) -> str:
    """Make a string safe for filenames."""
    name = str(name).strip()
    name = re.sub(r"[^A-Za-z0-9]+", "_", name)
    return name or "ligand"


def make_yamls_from_yaml_template_and_csv(
    yaml_template_path: str | Path,
    csv_path: str | Path,
    output_dir: str | Path,
    *,
    smiles_col: str = "smiles",
    name_col: str | None = "molecule_id",
    target_name: str = "target",
    ligand_id: str = "B",
) -> list[Path]:
    """
    Create one Boltz-2 YAML file per ligand in a CSV using an existing YAML template.
    
    Parameters
    ----------
    yaml_template_path : str or Path
        Path to existing YAML template file.
    csv_path : str or Path
        CSV containing ligands.
    output_dir : str or Path
        Directory to write YAML files.
    smiles_col : str
        Column name containing SMILES.
    name_col : str or None
        Optional column name for ligand identifiers.
    target_name : str
        Base name for YAML files.
    ligand_id : str
        ID of the ligand chain in the template (default: "B").
        
    Returns
    -------
    list[Path]
        Paths to generated YAML files.
    """
    yaml_template_path = Path(yaml_template_path)
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not yaml_template_path.exists():
        raise FileNotFoundError(f"YAML template not found: {yaml_template_path}")
    
    # Load template
    with open(yaml_template_path, "r") as f:
        template_data = yaml.safe_load(f)
    
    df = pd.read_csv(csv_path)
    
    if smiles_col not in df.columns:
        raise ValueError(
            f"CSV must contain a '{smiles_col}' column. "
            f"Found: {list(df.columns)}"
        )
    
    yaml_paths: list[Path] = []
    
    for i, row in df.iterrows():
        smiles = str(row[smiles_col]).strip()
        
        if not smiles or smiles.lower() == "nan":
            continue
        
        if name_col and name_col in df.columns:
            lig_name = _sanitize_name(row[name_col])
        else:
            lig_name = f"L{i+1:04d}"
        
        # Create a copy of the template
        yaml_data = yaml.safe_load(yaml.dump(template_data))
        
        # Update the ligand SMILES in the sequences
        for seq_entry in yaml_data.get("sequences", []):
            if "ligand" in seq_entry and seq_entry["ligand"].get("id") == ligand_id:
                seq_entry["ligand"]["smiles"] = smiles
                break
        
        yaml_path = output_dir / f"{_sanitize_name(target_name)}_{lig_name}.yaml"
        
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        
        yaml_paths.append(yaml_path)
    
    if not yaml_paths:
        raise ValueError("No valid ligands found in CSV")
    
    return yaml_paths


def make_yamls_from_protein_and_csv(
    protein_sequence: str,
    csv_path: str | Path,
    output_dir: str | Path,
    *,
    smiles_col: str = "smiles",
    name_col: str | None = "molecule_id",
    target_name: str = "3pbq",
) -> list[Path]:
    """
    Create one Boltz-2 YAML file per ligand in a CSV.

    Parameters
    ----------
    protein_sequence : str
        Amino acid sequence of the protein.
    csv_path : str or Path
        CSV containing ligands.
    output_dir : str or Path
        Directory to write YAML files.
    smiles_col : str
        Column name containing SMILES.
    name_col : str or None
        Optional column name for ligand identifiers.
    target_name : str
        Base name for YAML files.

    Returns
    -------
    list[Path]
        Paths to generated YAML files.
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    if smiles_col not in df.columns:
        raise ValueError(
            f"CSV must contain a '{smiles_col}' column. "
            f"Found: {list(df.columns)}"
        )

    yaml_paths: list[Path] = []

    protein_id = "A"
    ligand_id = "B"

    for i, row in df.iterrows():
        smiles = str(row[smiles_col]).strip()

        if not smiles or smiles.lower() == "nan":
            continue

        if name_col and name_col in df.columns:
            lig_name = _sanitize_name(row[name_col])
        else:
            lig_name = f"L{i+1:04d}"

        yaml_data = {
            "version": 1,
            "sequences": [
                {
                    "protein": {
                        "id": protein_id,
                        "sequence": protein_sequence,
                        # "msa": "/absolute/path/to/msa.a3m"  # optional
                    }
                },
                {
                    "ligand": {
                        "id": ligand_id,
                        "smiles": smiles,
                    }
                },
            ],
            "properties": [
                {
                    "affinity": {
                        "binder": ligand_id
                    }
                }
            ],
        }

        yaml_path = output_dir / f"{_sanitize_name(target_name)}_{lig_name}.yaml"

        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

        yaml_paths.append(yaml_path)

    if not yaml_paths:
        raise ValueError("No valid ligands found in CSV")

    return yaml_paths


import subprocess
import json
from pathlib import Path
from typing import List


def run_boltz_scoring(
    ligands_csv_path: str | Path,
    output_dir: str | Path,
    yamls_dir: str | Path,
    protein_pdb_path: str | None = None,
    protein_sequence: str | None = None,  # optional, if not provided, will be extracted from pdb file   
    yaml_template_path: str | Path | None = None,  # optional, if provided, use this template instead
    
    *,
    accelerator: str = "gpu",
    target_name: str | None = None,  # Optional target name for YAML files
    smiles_col: str = "smiles",
    name_col: str | None = "molecule_id",
    ligand_id: str = "B",  # ID of ligand chain in template
) -> list[float]:
    ligands_csv_path = Path(ligands_csv_path)
    output_dir = Path(output_dir)

    # 1. Generate YAMLs
    if yaml_template_path is not None:
        # Use YAML template
        yaml_paths = make_yamls_from_yaml_template_and_csv(
            yaml_template_path,
            ligands_csv_path,
            Path(yamls_dir),
            smiles_col=smiles_col,
            name_col=name_col,
            target_name=target_name or "target",
            ligand_id=ligand_id,
        )
    else:
        # Generate from PDB/sequence
        if protein_pdb_path is not None:
            protein_sequence = get_protein_sequence_from_pdb(protein_pdb_path)
        if protein_sequence is None:
            raise ValueError("Either yaml_template_path, protein_pdb_path, or protein_sequence must be provided")
        
        yaml_paths = make_yamls_from_protein_and_csv(
            protein_sequence,
            ligands_csv_path,
            Path(yamls_dir),
            smiles_col=smiles_col,
            name_col=name_col,
            target_name=target_name or "3pbq",
        )

    # 2. Run Boltz prediction
    # Convert Path objects to strings for subprocess
    yamls_dir_str = str(Path(yamls_dir).resolve())
    output_dir_str = str(output_dir.resolve())
    
    # Use poetry run boltz if pyproject.toml exists (we're in a poetry project)
    import os
    from pathlib import Path as PathLib
    
    # Check if pyproject.toml exists in current or parent directories
    current_dir = PathLib.cwd()
    has_poetry = (current_dir / "pyproject.toml").exists() or \
                 (current_dir.parent / "pyproject.toml").exists() or \
                 (current_dir.parent.parent / "pyproject.toml").exists()
    
    if has_poetry:
        cmd = ["poetry", "run", "boltz", "predict", yamls_dir_str, 
               "--accelerator", accelerator, 
               "--use_msa_server", 
               "--out_dir", output_dir_str]
    else:
        cmd = ["boltz", "predict", yamls_dir_str, 
               "--accelerator", accelerator, 
               "--use_msa_server", 
               "--out_dir", output_dir_str]
    
    # Run with real-time output streaming
    print(f"\nRunning Boltz prediction...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"\n✗ Boltz command failed with return code {result.returncode}")
        result.check_returncode()  # This will raise CalledProcessError

    # 3. Extract affinity scores
    scores: list[float] = []
    
    # Find the boltz_results directory
    boltz_results_dir = None
    for item in output_dir.iterdir():
        if item.is_dir() and item.name.startswith("boltz_results"):
            boltz_results_dir = item
            break
    
    if boltz_results_dir is None:
        raise FileNotFoundError(f"No boltz_results directory found in {output_dir}")
    
    predictions_dir = boltz_results_dir / "predictions"
    if not predictions_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")

    for yaml_path in yaml_paths:
        name = yaml_path.stem

        # Boltz output layout:
        # <output_dir>/boltz_results_<yamls_dir_name>/predictions/<sample_name>/affinity_<sample_name>.json
        # Sample name matches the YAML stem
        sample_dir = predictions_dir / name
        if not sample_dir.exists():
            # Try alternative naming: sometimes sample name is just the base name
            # Check if there's a directory that contains this name
            matching_dirs = [d for d in predictions_dir.iterdir() if d.is_dir() and name in d.name]
            if matching_dirs:
                sample_dir = matching_dirs[0]
            else:
                print(f"⚠ Warning: No prediction directory found for {name}, skipping...")
                continue  # Skip this molecule instead of failing
        
        # Look for affinity JSON file
        affinity_json = sample_dir / f"affinity_{sample_dir.name}.json"
        if not affinity_json.exists():
            # Try alternative naming
            json_files = list(sample_dir.glob("affinity_*.json"))
            if json_files:
                affinity_json = json_files[0]
            else:
                print(f"⚠ Warning: No affinity JSON found for {name} in {sample_dir}, skipping...")
                continue  # Skip this molecule instead of failing

        try:
            with open(affinity_json) as f:
                data = json.load(f)

            # Extract affinity value
            if "affinity_pred_value" in data:
                score = float(data["affinity_pred_value"])
            elif "affinity" in data:
                score = float(data["affinity"])
            elif "score" in data:
                score = float(data["score"])
            else:
                print(f"⚠ Warning: No affinity field found in {affinity_json} (keys: {list(data.keys())}), skipping...")
                continue  # Skip this molecule instead of failing

            scores.append(score)
        except Exception as e:
            print(f"⚠ Warning: Error processing {name}: {e}, skipping...")
            continue  # Skip this molecule instead of failing

    if not scores:
        raise ValueError(
            f"No valid scores extracted from {len(yaml_paths)} YAML files. "
            f"Check Boltz predictions in {predictions_dir}"
        )
    
    print(f"✓ Successfully extracted {len(scores)}/{len(yaml_paths)} affinity scores")
    return scores


def main():
    """Command-line interface for boltz_scoring."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate Boltz YAMLs and run affinity predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using a PDB file:
  python boltz_scoring.py --protein-pdb protein.pdb --ligands ligands.csv --output-dir outputs/
  
  # Using a protein sequence directly:
  python boltz_scoring.py --protein-sequence "MKTAYIAKQR..." --ligands ligands.csv --output-dir outputs/
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--ligands",
        type=Path,
        required=True,
        help="Path to CSV file containing ligands (must have 'smiles' column)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for Boltz predictions"
    )
    parser.add_argument(
        "--yamls-dir",
        type=Path,
        required=True,
        help="Directory to write YAML files"
    )
    
    # Protein input (either PDB, sequence, or YAML template)
    protein_group = parser.add_mutually_exclusive_group(required=False)
    protein_group.add_argument(
        "--protein-pdb",
        type=Path,
        help="Path to PDB file (protein sequence will be extracted)"
    )
    protein_group.add_argument(
        "--protein-sequence",
        type=str,
        help="Protein sequence as a string (1-letter amino acid codes)"
    )
    protein_group.add_argument(
        "--yaml-template",
        type=Path,
        help="Path to existing YAML template file (will replace ligand SMILES from CSV)"
    )
    
    parser.add_argument(
        "--ligand-id",
        default="B",
        help="ID of ligand chain in YAML template (default: B)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--accelerator",
        choices=["cpu", "gpu"],
        default="gpu",
        help="Hardware accelerator (default: gpu)"
    )
    parser.add_argument(
        "--smiles-col",
        default="smiles",
        help="Column name containing SMILES (default: smiles)"
    )
    parser.add_argument(
        "--name-col",
        default=None,
        help="Column name for ligand identifiers (optional)"
    )
    parser.add_argument(
        "--target-name",
        default="target",
        help="Base name for YAML files (default: target)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.ligands.exists():
        parser.error(f"Ligands CSV not found: {args.ligands}")
    
    if args.protein_pdb and not args.protein_pdb.exists():
        parser.error(f"PDB file not found: {args.protein_pdb}")
    
    # Create output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.yamls_dir.mkdir(parents=True, exist_ok=True)
    
    # Run scoring
    try:
        scores = run_boltz_scoring(
            ligands_csv_path=args.ligands,
            output_dir=args.output_dir,
            yamls_dir=args.yamls_dir,
            protein_pdb_path=args.protein_pdb,
            protein_sequence=args.protein_sequence,
            accelerator=args.accelerator,
        )
        
        print(f"\n✓ Successfully scored {len(scores)} ligand(s)")
        print(f"  Output directory: {args.output_dir}")
        print(f"  YAMLs directory: {args.yamls_dir}")
        print(f"\nAffinity scores:")
        for i, score in enumerate(scores, 1):
            print(f"  Ligand {i}: {score:.4f}")
        
        # Save scores to CSV
        scores_df = pd.DataFrame({
            "ligand_index": range(1, len(scores) + 1),
            "affinity_score": scores
        })
        scores_path = args.output_dir / "affinity_scores.csv"
        scores_df.to_csv(scores_path, index=False)
        print(f"\n✓ Scores saved to: {scores_path}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    import sys
    main()

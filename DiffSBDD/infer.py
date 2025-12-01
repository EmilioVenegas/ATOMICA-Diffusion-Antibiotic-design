#!/usr/bin/env python3
"""Inference script for cross-attention DiffSBDD model with evaluation."""

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from openbabel import openbabel
openbabel.obErrorLog.StopLogging()

import utils
from lightning_modules import LigandPocketDDPM
from analysis.metrics import BasicMolecularMetrics, MoleculeProperties
from analysis.molecule_builder import build_molecule


def prepare_data(
    pdbfile: str,
    resi_list: Optional[list[str]] = None,
    ref_ligand: Optional[str] = None,
) -> dict:
    """Prepare pocket data from PDB file.
    
    Args:
        pdbfile: Path to PDB file
        resi_list: List of residue identifiers (e.g., ['A:123', 'A:124'])
        ref_ligand: Path to reference ligand SDF or PDB format
        
    Returns:
        Dictionary with pocket information
    """
    print("=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)
    print(f"PDB file: {pdbfile}")
    
    if resi_list:
        print(f"Residue list: {resi_list}")
    elif ref_ligand:
        print(f"Reference ligand: {ref_ligand}")
    else:
        raise ValueError("Must provide either resi_list or ref_ligand")
    
    # The actual pocket preparation happens in model.generate_ligands
    # This function is a placeholder for data validation/preprocessing
    pocket_info = {
        'pdbfile': pdbfile,
        'resi_list': resi_list,
        'ref_ligand': ref_ligand,
    }
    
    print("Data preparation complete.")
    print()
    return pocket_info


def load_model(checkpoint: Path, device: str = 'cuda') -> LigandPocketDDPM:
    """Load trained model from checkpoint.
    
    Args:
        checkpoint: Path to model checkpoint (.ckpt file)
        device: Device to load model on ('cuda' or 'cpu')
        
    Returns:
        Loaded LigandPocketDDPM model
    """
    print("=" * 60)
    print("MODEL LOADING")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint}")
    print(f"Device: {device}")
    
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    
    model = LigandPocketDDPM.load_from_checkpoint(
        str(checkpoint), 
        map_location=device,
        strict=False  # Allow missing keys for flexibility
    )
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully.")
    print(f"Dataset: {model.dataset_name}")
    print(f"Pocket representation: {model.pocket_representation}")
    print()
    return model


def run_inference(
    model: LigandPocketDDPM,
    pdbfile: str,
    n_samples: int,
    batch_size: int,
    resi_list: Optional[list[str]] = None,
    ref_ligand: Optional[str] = None,
    num_nodes_lig: Optional[int] = None,
    sanitize: bool = False,
    all_frags: bool = False,
    relax: bool = False,
    resamplings: int = 10,
    jump_length: int = 1,
    timesteps: Optional[int] = None,
) -> list:
    """Run model inference to generate ligands.
    
    Args:
        model: Loaded LigandPocketDDPM model
        pdbfile: Path to PDB file
        n_samples: Number of ligand samples to generate
        batch_size: Batch size for generation
        resi_list: List of residue identifiers
        ref_ligand: Path to reference ligand
        num_nodes_lig: Fixed number of ligand nodes (optional)
        sanitize: Whether to sanitize molecules
        all_frags: Keep all fragments (vs largest only)
        relax: Whether to relax molecules
        resamplings: Number of resampling steps
        jump_length: Jump length for resampling
        timesteps: Number of diffusion timesteps
        
    Returns:
        List of generated RDKit molecule objects
    """
    print("=" * 60)
    print("MODEL INFERENCE")
    print("=" * 60)
    print(f"Generating {n_samples} samples...")
    print(f"Batch size: {batch_size}")
    print(f"Resamplings: {resamplings}")
    print(f"Timesteps: {timesteps if timesteps else 'default'}")
    
    if num_nodes_lig is not None:
        num_nodes_lig_tensor = torch.ones(n_samples, dtype=int) * num_nodes_lig
    else:
        num_nodes_lig_tensor = None
    
    molecules = []
    n_batches = n_samples // batch_size
    
    for i in range(n_batches):
        print(f"Batch {i+1}/{n_batches}...")
        molecules_batch = model.generate_ligands(
            pdbfile,
            batch_size,
            resi_list,
            ref_ligand,
            num_nodes_lig_tensor,
            sanitize,
            largest_frag=not all_frags,
            relax_iter=(200 if relax else 0),
            resamplings=resamplings,
            jump_length=jump_length,
            timesteps=timesteps
        )
        molecules.extend(molecules_batch)
    
    print(f"Generated {len(molecules)} molecules.")
    print()
    return molecules


def evaluate_molecules(
    molecules: list,
    dataset_info: dict,
    output_dir: Optional[Path] = None,
) -> dict:
    """Evaluate generated molecules with various metrics.
    
    Args:
        molecules: List of RDKit molecule objects
        dataset_info: Dataset information dict from model
        output_dir: Optional directory to save evaluation results
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    # Filter out None molecules (failed builds)
    valid_molecules = [m for m in molecules if m is not None]
    print(f"Valid molecules: {len(valid_molecules)}/{len(molecules)}")
    
    if len(valid_molecules) == 0:
        print("Warning: No valid molecules to evaluate!")
        return {}
    
    # Initialize metric calculators
    ligand_metrics = BasicMolecularMetrics(dataset_info)
    molecule_properties = MoleculeProperties()
    
    # Compute basic metrics
    (validity, connectivity, uniqueness, novelty), (_, connected_mols) = \
        ligand_metrics.evaluate_rdmols(valid_molecules)
    
    # Compute molecular properties
    qed, sa, logp, lipinski, diversity = \
        molecule_properties.evaluate_mean(connected_mols)
    
    # Build results dictionary
    results = {
        'n_generated': len(molecules),
        'n_valid': len(valid_molecules),
        'validity': float(validity),
        'connectivity': float(connectivity),
        'uniqueness': float(uniqueness),
        'novelty': float(novelty),
        'qed': float(qed),
        'sa_score': float(sa),
        'logp': float(logp),
        'lipinski_violations': float(lipinski),
        'diversity': float(diversity),
    }
    
    # Print results
    print(f"Validity: {validity:.3f}")
    print(f"Connectivity: {connectivity:.3f}")
    print(f"Uniqueness: {uniqueness:.3f}")
    print(f"Novelty: {novelty:.3f}")
    print(f"QED: {qed:.3f}")
    print(f"SA Score: {sa:.3f}")
    print(f"LogP: {logp:.3f}")
    print(f"Lipinski violations: {lipinski:.3f}")
    print(f"Diversity: {diversity:.3f}")
    
    # Save results if output directory provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "evaluation_results.json"
        with results_path.open('w') as f:
            json.dump(results, f, indent=2)
        print(f"\nEvaluation results saved to: {results_path}")
    
    print()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with cross-attention DiffSBDD model and evaluate results."
    )
    
    # Required arguments
    parser.add_argument('checkpoint', type=Path, help='Path to model checkpoint (.ckpt)')
    parser.add_argument('--pdbfile', type=str, required=True, help='Path to protein PDB file')
    
    # Pocket definition (mutually exclusive)
    pocket_group = parser.add_mutually_exclusive_group(required=True)
    pocket_group.add_argument('--resi_list', type=str, nargs='+', help='Residue list (e.g., A:123 A:124)')
    pocket_group.add_argument('--ref_ligand', type=str, help='Path to reference ligand SDF/PDB')
    
    # Output
    parser.add_argument('--outfile', type=Path, required=True, help='Output SDF file path')
    parser.add_argument('--eval_dir', type=Path, help='Directory to save evaluation results')
    
    # Generation parameters
    parser.add_argument('--n_samples', type=int, default=20, help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (default: n_samples)')
    parser.add_argument('--num_nodes_lig', type=int, default=None, help='Fixed number of ligand nodes')
    parser.add_argument('--resamplings', type=int, default=10, help='Number of resampling steps')
    parser.add_argument('--jump_length', type=int, default=1, help='Jump length for resampling')
    parser.add_argument('--timesteps', type=int, default=None, help='Number of diffusion timesteps')
    
    # Post-processing
    parser.add_argument('--sanitize', action='store_true', help='Sanitize molecules')
    parser.add_argument('--all_frags', action='store_true', help='Keep all fragments (vs largest only)')
    parser.add_argument('--relax', action='store_true', help='Relax molecules with force field')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Set batch size
    if args.batch_size is None:
        args.batch_size = args.n_samples
    assert args.n_samples % args.batch_size == 0, "n_samples must be divisible by batch_size"
    
    # ============================================================
    # 1. DATA PREPARATION
    # ============================================================
    pocket_info = prepare_data(
        args.pdbfile,
        args.resi_list,
        args.ref_ligand,
    )
    
    # ============================================================
    # 2. MODEL LOADING
    # ============================================================
    model = load_model(args.checkpoint, device)
    
    # ============================================================
    # 3. INFERENCE
    # ============================================================
    molecules = run_inference(
        model,
        args.pdbfile,
        args.n_samples,
        args.batch_size,
        args.resi_list,
        args.ref_ligand,
        args.num_nodes_lig,
        args.sanitize,
        args.all_frags,
        args.relax,
        args.resamplings,
        args.jump_length,
        args.timesteps,
    )
    
    # ============================================================
    # 4. SAVE RESULTS
    # ============================================================
    print("=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    utils.write_sdf_file(args.outfile, molecules)
    print(f"Molecules saved to: {args.outfile}")
    print()
    
    # ============================================================
    # 5. EVALUATION
    # ============================================================
    eval_dir = args.eval_dir or args.outfile.parent
    results = evaluate_molecules(
        molecules,
        model.dataset_info,
        eval_dir,
    )
    
    print("=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()


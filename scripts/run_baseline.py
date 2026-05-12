"""
Generate N molecules × M pockets from a DiffSBDD checkpoint.

Works directly from preprocessed .pt test files — no PDB files required.
The pocket dict is built from pre-computed tensors, so generation is bit-exact
with training preprocessing (same atom vocabulary, same CoM centering).

Usage (from project root):
    python scripts/run_baseline.py \\
        --checkpoint checkpoints/crossdocked_fullatom_cond.ckpt \\
        --test_dir DiffSBDD/data/processed_expert_atomica/test \\
        --outdir results/baseline_A \\
        --n_pockets 100 --n_samples 100 --batch_size 20 --timesteps 100 \\
        --no_atomica

    # With ATOMICA embeddings (conditions B/C/D):
    python scripts/run_baseline.py \\
        --checkpoint checkpoints/my_finetuned.ckpt \\
        --test_dir DiffSBDD/data/processed_expert_atomica/test \\
        --outdir results/cond_C

Outputs:
    <outdir>/complex_000001.sdf  (n_samples molecules per pocket)
    ...
    <outdir>/complex_000100.sdf
    <outdir>/meta.json           (run config + per-pocket molecule counts)
"""

import argparse
import json
import sys
import os
import warnings
from pathlib import Path

# Suppress OpenBabel stderr before the import triggers it
from openbabel import openbabel
openbabel.obErrorLog.StopLogging()

# Suppress RDKit valence / sanitization stderr
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean
from tqdm import tqdm

# Make DiffSBDD/ and project root importable regardless of cwd
_ROOT = Path(__file__).resolve().parent.parent
_DIFFSBDD = _ROOT / 'DiffSBDD'
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_DIFFSBDD))

import utils
from constants import FLOAT_TYPE, INT_TYPE
from lightning_modules import LigandPocketDDPM
from analysis.molecule_builder import build_molecule, process_molecule


def build_pocket_dict(data, n_samples, device, use_atomica=True):
    """
    Construct the pocket dict that ddpm.sample_given_pocket() expects,
    tiled n_samples times, from a single deserialized .pt record.
    """
    pocket_coord = data['pocket_coords'].to(device, FLOAT_TYPE)   # [N, 3]
    pocket_oh    = data['pocket_one_hot'].to(device, FLOAT_TYPE)  # [N, C]
    N = pocket_coord.shape[0]

    pocket_size = torch.tensor([N] * n_samples, device=device, dtype=INT_TYPE)
    pocket_mask = torch.repeat_interleave(
        torch.arange(n_samples, device=device, dtype=INT_TYPE), N
    )

    pocket = {
        'x':      pocket_coord.repeat(n_samples, 1),
        'one_hot': pocket_oh.repeat(n_samples, 1),
        'size':   pocket_size,
        'mask':   pocket_mask,
    }

    if use_atomica and 'pocket_atomica_embeddings' in data:
        emb = data['pocket_atomica_embeddings'].to(device, FLOAT_TYPE)  # [N, 32]
        pocket['atomica_embeddings'] = emb.repeat(n_samples, 1)

    return pocket


def generate_from_pocket(model, pocket, n_samples, timesteps, sanitize=True):
    """
    Run ddpm.sample_given_pocket() and return a list of valid RDKit molecules.
    """
    ddpm = model.ddpm
    ddpm.eval()

    with torch.no_grad():
        num_nodes_lig = ddpm.size_distribution.sample_conditional(
            n1=None, n2=pocket['size'])

        pocket_com_before = scatter_mean(
            pocket['x'], pocket['mask'], dim=0)

        xh_lig, xh_pocket, lig_mask, pocket_mask = ddpm.sample_given_pocket(
            pocket, num_nodes_lig, timesteps=timesteps)

        # Translate generated ligand back to original pocket CoM
        pocket_com_after = scatter_mean(
            xh_pocket[:, :model.x_dims], pocket_mask, dim=0)
        shift = (pocket_com_before - pocket_com_after)[pocket_mask]
        xh_pocket[:, :model.x_dims] += shift
        shift = (pocket_com_before - pocket_com_after)[lig_mask]
        xh_lig[:, :model.x_dims] += shift

    x         = xh_lig[:, :model.x_dims].detach().cpu()
    atom_type = xh_lig[:, model.x_dims:].argmax(1).detach().cpu()
    lig_mask  = lig_mask.cpu()

    molecules = []
    for coords, types in zip(
            utils.batch_to_list(x, lig_mask),
            utils.batch_to_list(atom_type, lig_mask)):
        mol = build_molecule(coords, types, model.dataset_info, add_coords=True)
        mol = process_molecule(mol, sanitize=sanitize, largest_frag=True,
                               add_hydrogens=False, relax_iter=0)
        if mol is not None:
            molecules.append(mol)

    return molecules


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--test_dir',   type=Path,
                   default='DiffSBDD/data/processed_expert_atomica/test')
    p.add_argument('--outdir',     type=Path, required=True)
    p.add_argument('--n_pockets',  type=int, default=100)
    p.add_argument('--n_samples',  type=int, default=100)
    p.add_argument('--batch_size', type=int, default=20,
                   help='Pockets processed simultaneously (memory vs. speed)')
    p.add_argument('--timesteps',  type=int, default=100)
    p.add_argument('--no_atomica', action='store_true',
                   help='Ignore pre-computed ATOMICA embeddings (condition A)')
    p.add_argument('--sanitize',   action='store_true', default=True)
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.checkpoint} ...")
    model = LigandPocketDDPM.load_from_checkpoint(
        str(args.checkpoint), map_location=device,
        adapter_lr=1e-4, freeze_backbone=False, strict=False)
    model = model.to(device)
    model.eval()

    test_files = sorted(args.test_dir.glob('complex_*.pt'))[:args.n_pockets]
    if len(test_files) == 0:
        raise FileNotFoundError(f"No .pt files found in {args.test_dir}")
    print(f"Generating {args.n_samples} molecules for each of "
          f"{len(test_files)} pockets (timesteps={args.timesteps}, "
          f"atomica={'off' if args.no_atomica else 'on'}) ...")

    meta = {
        'checkpoint': str(args.checkpoint),
        'n_pockets':  len(test_files),
        'n_samples':  args.n_samples,
        'timesteps':  args.timesteps,
        'no_atomica': args.no_atomica,
        'pockets': []
    }

    assert args.n_samples % args.batch_size == 0, \
        f"n_samples ({args.n_samples}) must be divisible by batch_size ({args.batch_size})"
    n_batches = args.n_samples // args.batch_size

    for pocket_idx, pt_file in enumerate(tqdm(test_files, desc='Pockets')):
        data = torch.load(pt_file, map_location='cpu')
        pocket_name = pt_file.stem   # e.g. complex_000001 — matches PDB extractor naming
        sdf_path = args.outdir / f"{pocket_name}.sdf"

        all_mols = []
        for _ in range(n_batches):
            pocket = build_pocket_dict(
                data, args.batch_size, device,
                use_atomica=not args.no_atomica)
            mols = generate_from_pocket(
                model, pocket, args.batch_size,
                timesteps=args.timesteps, sanitize=args.sanitize)
            all_mols.extend(mols)

        utils.write_sdf_file(sdf_path, all_mols)

        meta['pockets'].append({
            'pocket_name': pocket_name,
            'source_file': pt_file.name,
            'source_id':   str(data.get('name', pocket_idx)),
            'n_generated': len(all_mols),
            'sdf_file':    sdf_path.name,
        })

    with open(args.outdir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    total_mols = sum(p['n_generated'] for p in meta['pockets'])
    print(f"\nDone. {total_mols} molecules written to {args.outdir}/")
    print(f"Validity rate: {total_mols / (len(test_files) * args.n_samples):.1%}")


if __name__ == '__main__':
    main()

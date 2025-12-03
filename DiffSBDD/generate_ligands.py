import argparse
from pathlib import Path

import torch
from openbabel import openbabel
openbabel.obErrorLog.StopLogging()  # suppress OpenBabel messages

import utils
from lightning_modules import LigandPocketDDPM

import sys
import os
# Ensure root is in path for ATOMICA imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ATOMICA.models.prediction_model import PredictionModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--pdbfile', type=str)
    parser.add_argument('--resi_list', type=str, nargs='+', default=None)
    parser.add_argument('--ref_ligand', type=str, default=None)
    parser.add_argument('--outfile', type=Path)
    parser.add_argument('--n_samples', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_nodes_lig', type=int, default=None)
    parser.add_argument('--all_frags', action='store_true')
    parser.add_argument('--sanitize', action='store_true')
    parser.add_argument('--relax', action='store_true')
    parser.add_argument('--resamplings', type=int, default=10)
    parser.add_argument('--jump_length', type=int, default=1)
    parser.add_argument('--timesteps', type=int, default=None)
    parser.add_argument('--atomica_config', type=str, default='ATOMICA/pretrain/pretrain_model_config.json')
    parser.add_argument('--atomica_weights', type=str, default='ATOMICA/pretrain/pretrain_model_weights.pt')
    args = parser.parse_args()

    pdb_id = Path(args.pdbfile).stem

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.batch_size is None:
        args.batch_size = args.n_samples
    assert args.n_samples % args.batch_size == 0

    # Load model
    model = LigandPocketDDPM.load_from_checkpoint(
        args.checkpoint, map_location=device)
    model = model.to(device)

    # Load ATOMICA model if needed
    atomica_model = None
    if getattr(model.hparams.egnn_params, 'atomica_nf', 0) > 0:
        print("Loading ATOMICA model...")
        try:
            atomica_model = PredictionModel.load_from_config_and_weights(args.atomica_config, args.atomica_weights).to(device).eval()
            print("ATOMICA model loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load ATOMICA model: {e}")
            print("Generation might fail if the model expects embeddings.")

    if args.num_nodes_lig is not None:
        num_nodes_lig = torch.ones(args.n_samples, dtype=int) * \
                        args.num_nodes_lig
    else:
        num_nodes_lig = None

    molecules = []
    for i in range(args.n_samples // args.batch_size):
        molecules_batch = model.generate_ligands(
            args.pdbfile, args.batch_size, args.resi_list, args.ref_ligand,
            num_nodes_lig, args.sanitize, largest_frag=not args.all_frags,
            relax_iter=(200 if args.relax else 0),
            resamplings=args.resamplings, jump_length=args.jump_length,
            timesteps=args.timesteps, atomica_model=atomica_model)
        molecules.extend(molecules_batch)

    # Make SDF files
    utils.write_sdf_file(args.outfile, molecules)

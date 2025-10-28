"""
data_prep_pdbbind.py

PDBBind dataset preparation script for transfer learning.
Downloads or points to local PDBBind directory.
Logs versions/subsets used (e.g., v2015, v2020, refined/core).

Usage:
    python data_prep_pdbbind.py --subset refined --version v2015
    python data_prep_pdbbind.py --subset core --version v2015 --local_path /path/to/pdbbind
    python data_prep_pdbbind.py --help

Prerequisites:
    - DGL-LifeSci installed (pip install -e python/)
    - RDKit installed (conda install -c conda-forge rdkit OR pip install rdkit-pypi)
    - If using local PDBBind, ensure it follows the v2015 structure/naming format
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Prepare PDBBind dataset for binding affinity transfer learning'
    )
    
    parser.add_argument(
        '--subset',
        type=str,
        choices=['refined', 'core'],
        default='refined',
        help='PDBBind subset to load (refined or core). Default: refined'
    )
    
    parser.add_argument(
        '--version',
        type=str,
        choices=['v2007', 'v2015', 'v2020'],
        default='v2015',
        help='PDBBind version. Default: v2015. Note: v2020 may require custom loading.'
    )
    
    parser.add_argument(
        '--local_path',
        type=str,
        default=None,
        help='Local path to existing PDBBind dataset. If not provided, will download from DGL database.'
    )
    
    parser.add_argument(
        '--load_binding_pocket',
        action='store_true',
        default=True,
        help='Load binding pockets instead of full proteins. Default: True'
    )
    
    parser.add_argument(
        '--remove_coreset_from_refinedset',
        action='store_true',
        default=True,
        help='Remove core set from refined set (useful when training on refined, testing on core). Default: True'
    )
    
    parser.add_argument(
        '--sanitize',
        action='store_true',
        default=False,
        help='Perform sanitization in RDKit molecule initialization. Default: False'
    )
    
    parser.add_argument(
        '--num_processes',
        type=int,
        default=None,
        help='Number of worker processes for loading. Default: number of CPUs'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data',
        help='Directory to save dataset metadata and logs. Default: ./data'
    )
    
    return parser.parse_args()


def log_dataset_info(dataset, args):
    """
    Log information about the loaded dataset.
    
    Args:
        dataset: PDBBind dataset instance
        args: Command line arguments
    """
    logger.info("=" * 60)
    logger.info("PDBBind Dataset Loaded Successfully")
    logger.info("=" * 60)
    logger.info(f"Subset: {args.subset}")
    logger.info(f"Version: {args.version}")
    logger.info(f"Local path: {args.local_path if args.local_path else 'Downloaded from DGL'}")
    logger.info(f"Number of complexes: {len(dataset)}")
    logger.info(f"Task names: {dataset.task_names}")
    logger.info(f"Number of tasks: {dataset.n_tasks}")
    
    # TODO: Add statistics about the dataset
    # - Affinity range (min, max, mean, std)
    # - Ligand size distribution
    # - Protein size distribution
    # - Missing values if any
    
    logger.info("=" * 60)


def save_dataset_metadata(dataset, args, output_dir):
    """
    Save dataset metadata to file for reproducibility.
    
    Args:
        dataset: PDBBind dataset instance
        args: Command line arguments
        output_dir: Directory to save metadata
    """
    import json
    from datetime import datetime
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'subset': args.subset,
        'version': args.version,
        'local_path': args.local_path,
        'load_binding_pocket': args.load_binding_pocket,
        'remove_coreset_from_refinedset': args.remove_coreset_from_refinedset,
        'sanitize': args.sanitize,
        'num_complexes': len(dataset),
        'task_names': dataset.task_names,
        'n_tasks': dataset.n_tasks,
    }
    
    # TODO: Add more metadata
    # - Affinity statistics
    # - Complex IDs list
    # - Train/val/test split information (if applicable)
    
    metadata_path = output_dir / f'pdbbind_{args.subset}_{args.version}_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata saved to: {metadata_path}")


def load_pdbbind_dataset(args):
    """
    Load PDBBind dataset using DGLLife.
    
    Args:
        args: Command line arguments
        
    Returns:
        dataset: PDBBind dataset instance
    """
    try:
        from dgllife.data import PDBBind
        # TODO: Import appropriate featurization functions based on model choice
        # from dgllife.utils import ACNN_graph_construction_and_featurization
        # from dgllife.utils import PN_graph_construction_and_featurization
    except ImportError as e:
        logger.error("Failed to import DGLLife. Please install it first:")
        logger.error("  cd python && pip install -e .")
        sys.exit(1)
    
    logger.info("Loading PDBBind dataset...")
    logger.info(f"Subset: {args.subset}, Version: {args.version}")
    
    if args.local_path:
        logger.info(f"Using local PDBBind data from: {args.local_path}")
        if not os.path.exists(args.local_path):
            logger.error(f"Local path does not exist: {args.local_path}")
            sys.exit(1)
    else:
        logger.info("PDBBind will be downloaded from DGL database if not cached")
    
    # TODO: Choose appropriate featurization based on model (ACNN or PotentialNet)
    # For transfer learning, you may want to use PotentialNet featurization
    # construct_graph_fn = PN_graph_construction_and_featurization
    
    try:
        dataset = PDBBind(
            subset=args.subset,
            pdb_version=args.version,
            load_binding_pocket=args.load_binding_pocket,
            remove_coreset_from_refinedset=args.remove_coreset_from_refinedset,
            sanitize=args.sanitize,
            calc_charges=False,  # Set to True if needed for featurization
            remove_hs=False,     # Keep hydrogens for more accurate 3D structure
            use_conformation=True,  # Essential for 3D structure-based models
            # construct_graph_and_featurize=construct_graph_fn,  # TODO: Uncomment and set
            zero_padding=True,   # For consistent batch sizes
            num_processes=args.num_processes,
            local_path=args.local_path
        )
        
        logger.info(f"Successfully loaded {len(dataset)} protein-ligand complexes")
        return dataset
        
    except Exception as e:
        logger.error(f"Failed to load PDBBind dataset: {e}")
        logger.error("If using v2020, you may need to download it manually from:")
        logger.error("  http://www.pdbbind.org.cn/download.php")
        logger.error("And specify --local_path pointing to the extracted directory")
        raise


def download_instructions():
    """Print instructions for manually downloading PDBBind."""
    logger.info("\n" + "=" * 60)
    logger.info("PDBBind Dataset Download Instructions")
    logger.info("=" * 60)
    logger.info("\nFor PDBBind v2015 (automatically downloaded by DGL):")
    logger.info("  - No action needed, DGL will download and cache it")
    logger.info("\nFor PDBBind v2020 (latest, manual download required):")
    logger.info("  1. Visit: http://www.pdbbind.org.cn/download.php")
    logger.info("  2. Register and download the refined/core sets")
    logger.info("  3. Extract to a local directory")
    logger.info("  4. Run this script with --local_path /path/to/pdbbind")
    logger.info("\nPDBBind Structure (v2015 format expected):")
    logger.info("  PDBBind_v2015/")
    logger.info("    ├── refined-set/")
    logger.info("    │   ├── index/")
    logger.info("    │   │   └── INDEX_refined_data.2015")
    logger.info("    │   └── [PDB_ID]/")
    logger.info("    │       ├── [PDB_ID]_ligand.mol2")
    logger.info("    │       ├── [PDB_ID]_protein.pdb")
    logger.info("    │       └── [PDB_ID]_pocket.pdb")
    logger.info("    └── core-set/")
    logger.info("        └── (similar structure)")
    logger.info("=" * 60 + "\n")


def main():
    """Main execution function."""
    args = parse_args()
    
    logger.info("PDBBind Dataset Preparation for Transfer Learning")
    logger.info("=" * 60)
    
    # Print download instructions if needed
    if args.version == 'v2020' and not args.local_path:
        logger.warning("v2020 requires manual download. Printing instructions...")
        download_instructions()
        logger.error("Please download v2020 manually and re-run with --local_path")
        sys.exit(1)
    
    # Load dataset
    dataset = load_pdbbind_dataset(args)
    
    # Log dataset information
    log_dataset_info(dataset, args)
    
    # Save metadata
    save_dataset_metadata(dataset, args, args.output_dir)
    
    # TODO: Additional preprocessing steps
    # - Create train/val/test splits
    # - Compute and cache molecular descriptors
    # - Prepare batched data loaders
    # - Visualize affinity distribution
    
    logger.info("\nDataset preparation complete!")
    logger.info(f"You can now use this dataset for training with:")
    logger.info(f"  python train_transfer.py --subset {args.subset} --version {args.version}")
    
    return dataset


if __name__ == '__main__':
    main()

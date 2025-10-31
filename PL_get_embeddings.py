import gzip
import json
import sys
from collections import defaultdict
from tqdm import tqdm

# === NEW IMPORTS FOR EMBEDDING ===
import argparse
import pickle
import torch
from data.dataset import PDBDataset, ProtInterfaceDataset
from models.prediction_model import PredictionModel
from models.pretrain_model import DenoisePretrainModel
from models.prot_interface_model import ProteinInterfaceModel
from trainers.abs_trainer import Trainer
import numpy as np

# --- Configuration (kept for analysis path) ---
FILE_NAME = '/Users/shawnkang/ATOMICA/PL.jsonl.gz'


#HELLO THIS IS AN EDIT 
def choose_device():
    import torch
    # Prefer CUDA if you ever move to an NVIDIA box
    if torch.cuda.is_available():
        return "cuda"
    # torch-scatter on macOS doesn't support MPS -> force CPU
    return "cpu"


def analyze_dataset():
    """
    Reads the gzipped JSON Lines file to analyze its contents based on the
    provided sample structure.
    """
    print(f"--- STARTING FULL ANALYSIS of {FILE_NAME} ---")
    total_complexes = 0
    top_level_keys = defaultdict(int)
    data_sub_keys = defaultdict(int)
    protein_atom_counts, ligand_atom_counts, total_atom_counts = [], [], []
    affinity_values = []
    try:
        with gzip.open(FILE_NAME, 'rt', encoding='utf-8') as f:
            for line in tqdm(f, desc="Analyzing complexes"):
                try:
                    data = json.loads(line)
                    total_complexes += 1
                    for key in data.keys():
                        top_level_keys[key] += 1
                    p_atoms = 0
                    l_atoms = 0
                    main_data = data.get('data', {})
                    block_lengths = main_data.get('block_lengths', [])
                    segment_ids = main_data.get('segment_ids', [])
                    all_atoms_x = main_data.get('X', [])
                    total_atom_counts.append(len(all_atoms_x))
                    for key in main_data.keys():
                        data_sub_keys[key] += 1
                    if len(block_lengths) == len(segment_ids):
                        for length, seg_id in zip(block_lengths, segment_ids):
                            if seg_id == 0:
                                p_atoms += length
                            elif seg_id == 1:
                                l_atoms += length
                        if p_atoms > 0:
                            protein_atom_counts.append(p_atoms)
                        if l_atoms > 0:
                            ligand_atom_counts.append(l_atoms)
                    affinity_val = data.get('affinity', {}).get('neglog_aff')
                    if affinity_val is not None:
                        affinity_values.append(affinity_val)
                except json.JSONDecodeError:
                    print(f"\nWarning: Skipping a line that was not valid JSON.")
                except Exception as e:
                    print(f"\nWarning: Skipping line due to unexpected error: {e}")
    except FileNotFoundError:
        print(f"Error: File not found at '{FILE_NAME}'")
        return
    except Exception as e:
        print(f"An unexpected error occurred during analysis: {e}")
        return

    print("\n--- ANALYSIS COMPLETE ---")
    print(f"\nTotal complexes found: {total_complexes:,}")
    print("\nTop-level JSON keys found (and frequency):")
    for key, count in top_level_keys.items():
        print(f"  - '{key}': {count:,} times")
    print("\nSub-keys inside 'data' object (and frequency):")
    for key, count in data_sub_keys.items():
        print(f"  - '{key}': {count:,} times")

    if total_atom_counts:
        print(f"\n--- Total Atom Statistics (from 'data.X') ---")
        print(f"  Min atoms:    {min(total_atom_counts)}")
        print(f"  Max atoms:    {max(total_atom_counts)}")
        print(f"  Avg atoms:    {sum(total_atom_counts) / len(total_atom_counts):.2f}")

    if protein_atom_counts:
        print(f"\n--- Protein Atom Statistics (from 'segment_id == 0') ---")
        print(f"  Min atoms:    {min(protein_atom_counts)}")
        print(f"  Max atoms:    {max(protein_atom_counts)}")
        print(f"  Avg atoms:    {sum(protein_atom_counts) / len(protein_atom_counts):.2f}")

    if ligand_atom_counts:
        print(f"\n--- Ligand Atom Statistics (from 'segment_id == 1') ---")
        print(f"  Min atoms:    {min(ligand_atom_counts)}")
        print(f"  Max atoms:    {max(ligand_atom_counts)}")
        print(f"  Avg atoms:    {sum(ligand_atom_counts) / len(ligand_atom_counts):.2f}")

    if affinity_values:
        print(f"\n--- Affinity Statistics ('neglog_aff') ---")
        print(f"  Min affinity:    {min(affinity_values)}")
        print(f"  Max affinity:    {max(affinity_values)}")
        print(f"  Avg affinity:    {sum(affinity_values) / len(affinity_values):.2f}")

    if not protein_atom_counts or not ligand_atom_counts:
        print("\n*** WARNING: Could not find any protein/ligand atoms. ***")
        print("This might mean the 'segment_ids' are not 0 or 1, or the")
        print("'block_lengths' / 'segment_ids' keys were not found.")

# ===================== NEW: EMBEDDING PIPELINE =====================

def load_model(args):
    """Load ATOMICA model from checkpoint or (config + weights)."""
    if args.model_ckpt:
        model = torch.load(args.model_ckpt, map_location="cpu")
    elif args.model_config and args.model_weights:
        with open(args.model_config, "r") as f:
            model_config = json.load(f)
        if model_config['model_type'] in ('PredictionModel', 'DenoisePretrainModel'):
            model = PredictionModel.load_from_config_and_weights(args.model_config, args.model_weights)
        elif model_config['model_type'] == 'ProteinInterfaceModel':
            model = ProteinInterfaceModel.load_from_config_and_weights(args.model_config, args.model_weights)
        else:
            raise NotImplementedError(f"Model type {model_config['model_type']} not implemented")
    else:
        raise ValueError("Provide --model_ckpt OR (--model_config + --model_weights).")

    # If it's a ProteinInterfaceModel, unwrap to prot_model
    if isinstance(model, ProteinInterfaceModel):
        print("Model is ProteinInterfaceModel, extracting prot_model.")
        model = model.prot_model

    # If it's a pure pretrain model, wrap as PredictionModel for infer()
    if isinstance(model, DenoisePretrainModel) and not isinstance(model, PredictionModel):
        model = PredictionModel.load_from_pretrained(args.model_ckpt)

    device = choose_device()
    print(f"[info] device = {device}")
    model = model.to(device)
    model.eval()
    return model, device

def infer_batch(model, batch):
    """Run ATOMICA inference on one collated batch dict."""
    with torch.no_grad():
        return model.infer(batch)

def embed_dataset(
    data_path,
    output_path,
    model,
    batch_size=2,
    shard_size=5000,   # saves every N items to avoid huge RAM
    use_interface_dataset=False,
    device="cpu"
):
    """
    Stream PL.jsonl.gz and write embeddings to .pkl files.
    If shard_size is not None, writes multiple files: output_path.replace('.pkl', f'.part{n}.pkl')
    """
    # Choose dataset class (PL.jsonl.gz -> PDBDataset)
    dataset_class = ProtInterfaceDataset if use_interface_dataset else PDBDataset

    embeddings_buffer = []
    shard_idx = 0
    total_items = 0

    # Small helper to flush a shard to disk
    def flush_shard():
        nonlocal embeddings_buffer, shard_idx
        if not embeddings_buffer:
            return
        shard_out = output_path if shard_size is None else output_path.replace(".pkl", f".part{shard_idx}.pkl")
        with open(shard_out, "wb") as f:
            pickle.dump(embeddings_buffer, f)
        print(f"[saved] {len(embeddings_buffer)} items -> {shard_out}")
        embeddings_buffer = []
        shard_idx += 1

    # Stream the jsonl.gz
    with gzip.open(data_path, "rt", encoding="utf-8") as f:
        items = []          # raw json lines parsed
        batch_payload = []  # the "data" dicts given to collate_fn

        for line in tqdm(f, desc="Embedding complexes"):
            rec = json.loads(line)
            items.append(rec)
            if use_interface_dataset:
                batch_payload.append(rec["prot_data"])
            else:
                batch_payload.append(rec["data"])

            # When batch is full -> collate and infer
            if len(batch_payload) == batch_size:
                try:
                    batch = dataset_class.collate_fn(batch_payload)
                    batch = Trainer.to_device(batch, device)
                    ret = infer_batch(model, batch)

                    # unpack batch outputs
                    curr_block = 0
                    curr_atom = 0
                    for i, item in enumerate(items):
                        num_blocks = len(item["data"]["B"])
                        num_atoms = len(item["data"]["A"])
                        out = {
                            "id": item["id"],
                            "graph_embedding": ret.graph_repr[i].detach().cpu().numpy(),
                            "block_embedding": ret.block_repr[curr_block:curr_block+num_blocks].detach().cpu().numpy(),
                            "atom_embedding":  ret.unit_repr[curr_atom:curr_atom+num_atoms].detach().cpu().numpy(),
                            "block_id": item["data"]["B"],
                            "atom_id": item["data"]["A"],
                        }
                        embeddings_buffer.append(out)
                        curr_block += num_blocks
                        curr_atom  += num_atoms
                        total_items += 1

                    # flush shard if needed
                    if shard_size is not None and len(embeddings_buffer) >= shard_size:
                        flush_shard()

                except Exception as e:
                    # GPU OOM fallback: process one-by-one
                    if "CUDA out of memory" in str(e):
                        torch.cuda.empty_cache()
                        for item in items:
                            try:
                                single_batch = dataset_class.collate_fn(
                                    [item["prot_data"] if use_interface_dataset else item["data"]]
                                )
                                single_batch = Trainer.to_device(single_batch, device)
                                ret1 = infer_batch(model, single_batch)
                                out = {
                                    "id": item["id"],
                                    "graph_embedding": ret1.graph_repr[0].detach().cpu().numpy(),
                                    "block_embedding": ret1.block_repr.detach().cpu().numpy(),
                                    "atom_embedding":  ret1.unit_repr.detach().cpu().numpy(),
                                    "block_id": item["data"]["B"],
                                    "atom_id": item["data"]["A"],
                                }
                                embeddings_buffer.append(out)
                                total_items += 1
                                if shard_size is not None and len(embeddings_buffer) >= shard_size:
                                    flush_shard()
                            except Exception as e1:
                                print(f"[skip] {item.get('id')} due to error: {e1}")
                                torch.cuda.empty_cache()
                                continue
                    else:
                        print(f"[batch error] {e}")
                        raise

                # reset batch accumulators
                items = []
                batch_payload = []

        # handle tail (incomplete last batch)
        if batch_payload:
            try:
                batch = dataset_class.collate_fn(batch_payload)
                batch = Trainer.to_device(batch, device)
                ret = infer_batch(model, batch)
                curr_block = 0
                curr_atom = 0
                for i, item in enumerate(items):
                    num_blocks = len(item["data"]["B"])
                    num_atoms  = len(item["data"]["A"])
                    out = {
                        "id": item["id"],
                        "graph_embedding": ret.graph_repr[i].detach().cpu().numpy(),
                        "block_embedding": ret.block_repr[curr_block:curr_block+num_blocks].detach().cpu().numpy(),
                        "atom_embedding":  ret.unit_repr[curr_atom:curr_atom+num_atoms].detach().cpu().numpy(),
                        "block_id": item["data"]["B"],
                        "atom_id": item["data"]["A"],
                    }
                    embeddings_buffer.append(out)
                    total_items += 1
                    curr_block += num_blocks
                    curr_atom  += num_atoms
            except Exception as e:
                if "CUDA out of memory" in str(e):
                    torch.cuda.empty_cache()
                    for item in items:
                        try:
                            single_batch = dataset_class.collate_fn(
                                [item["prot_data"] if use_interface_dataset else item["data"]]
                            )
                            single_batch = Trainer.to_device(single_batch, device)
                            ret1 = infer_batch(model, single_batch)
                            out = {
                                "id": item["id"],
                                "graph_embedding": ret1.graph_repr[0].detach().cpu().numpy(),
                                "block_embedding": ret1.block_repr.detach().cpu().numpy(),
                                "atom_embedding":  ret1.unit_repr.detach().cpu().numpy(),
                                "block_id": item["data"]["B"],
                                "atom_id": item["data"]["A"],
                            }
                            embeddings_buffer.append(out)
                            total_items += 1
                        except Exception as e1:
                            print(f"[skip] {item.get('id')} due to error: {e1}")
                            torch.cuda.empty_cache()
                            continue
                else:
                    print(f"[tail batch error] {e}")
                    raise

    # final flush
    flush_shard()
    print(f"[done] total items embedded: {total_items}")

def parse_args():
    p = argparse.ArgumentParser()
    # mode
    p.add_argument("--mode", choices=["analyze", "embed"], default="embed",
                   help="analyze: print stats (your original). embed: write embeddings to disk.")
    # model loading
    p.add_argument("--model_ckpt", type=str, default=None, help="Path to ATOMICA checkpoint (.ckpt/.pt)")
    p.add_argument("--model_config", type=str, default=None, help="Optional JSON config (alternative to --model_ckpt)")
    p.add_argument("--model_weights", type=str, default=None, help="Optional weights file (alternative to --model_ckpt)")
    # data & IO
    p.add_argument("--data_path", type=str, default=FILE_NAME, help="Path to PL.jsonl.gz")
    p.add_argument("--output_path", type=str, default="/Users/shawnkang/ATOMICA/PL_embeddings.pkl",
                   help="Output .pkl (or shard base name)")
    # performance
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--shard_size", type=int, default=5000,
                   help="Save every N items; set to -1 to keep all in memory and write once")
    p.add_argument("--use_interface_dataset", action="store_true",
                   help="Use ProtInterfaceDataset instead of PDBDataset (usually not needed for PL.jsonl.gz)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.mode == "analyze":
        analyze_dataset()
    else:
        if args.shard_size is not None and args.shard_size < 0:
            shard_size = None
        else:
            shard_size = args.shard_size
        model, device = load_model(args)
        embed_dataset(
            data_path=args.data_path,
            output_path=args.output_path,
            model=model,
            batch_size=args.batch_size,
            shard_size=shard_size,
            use_interface_dataset=args.use_interface_dataset,
            device=device
        )

"""
Extract receptor PDB files for the test-set pockets from the CrossDocked LMDB.

Each test .pt file's 'name' field is a direct key into the LMDB.  We look up
the full atom-level protein structure (protein_pos, protein_atom_name,
protein_atom_to_aa_type, protein_atom2residue, amino_acid, res_idx) and write
a proper ATOM-record PDB that AutoDock Vina and PoseBusters can consume.

Also writes the reference ligand as an SDF file and a <pocket>.box.txt with
docking box parameters (center + size in Angstroms, 20 Å box).

Usage (from project root):
    python scripts/extract_pocket_pdbs.py \\
        --test_dir  DiffSBDD/data/processed_expert_atomica/test \\
        --lmdb_path data/crossdocked_pocket10_processed.lmdb \\
        --outdir    data/receptor_pdbs \\
        --n_pockets 100
"""

import argparse
import pickle
import sys
from pathlib import Path

import lmdb
import numpy as np
import torch
from tqdm import tqdm

# ─── Atom-name vocabulary (decoded from the LMDB encoding) ──────────────────
# Backbone: 1=N, 2=CA, 3=C, 4=O
# Cβ: 5=CB
# Cγ variants: 6=CG, 7=CG1, 8=CG2
# γ heteroatoms: 9=OG, 10=OG1, 11=SG
# Cδ: 12=CD, 13=CD1, 14=CD2
# Nδ: 15=ND1, 16=ND2
# Oδ: 17=OD1, 18=OD2
# Sδ: 19=SD
# Cε: 20=CE, 21=CE1, 22=CE2, 23=CE3
# Nε: 24=NE, 25=NE1, 26=NE2
# Oε: 27=OE1, 28=OE2
# Cζ/misc: 29=CZ2, 30=NH1, 31=NH2, 32=OH, 33=CZ, 34=CZ3, 35=CH2
# Nζ: 36=NZ
ATOM_NAME_VOCAB = {
    1: 'N',   2: 'CA',  3: 'C',   4: 'O',   5: 'CB',
    6: 'CG',  7: 'CG1', 8: 'CG2', 9: 'OG',  10: 'OG1',
    11: 'SG', 12: 'CD', 13: 'CD1', 14: 'CD2', 15: 'ND1',
    16: 'ND2', 17: 'OD1', 18: 'OD2', 19: 'SD', 20: 'CE',
    21: 'CE1', 22: 'CE2', 23: 'CE3', 24: 'NE', 25: 'NE1',
    26: 'NE2', 27: 'OE1', 28: 'OE2', 29: 'CZ2', 30: 'NH1',
    31: 'NH2', 32: 'OH', 33: 'CZ', 34: 'CZ3', 35: 'CH2',
    36: 'NZ',
}

# 1-indexed alphabetical amino acid encoding used in the LMDB
AA_VOCAB = {
    1: 'ALA', 2: 'ARG', 3: 'ASN', 4: 'ASP', 5: 'CYS',
    6: 'GLN', 7: 'GLU', 8: 'GLY', 9: 'HIS', 10: 'ILE',
    11: 'LEU', 12: 'LYS', 13: 'MET', 14: 'PHE', 15: 'PRO',
    16: 'SER', 17: 'THR', 18: 'TRP', 19: 'TYR', 20: 'VAL',
}

ELEMENT_SYMBOL = {6: 'C', 7: 'N', 8: 'O', 16: 'S'}


def _pdb_atom_line(serial, atom_name, res_name, chain, res_seq, x, y, z, element):
    """Return a PDB ATOM record string (80 chars)."""
    # PDB atom name: 4-char field; ≤3-char names are right-padded after col 14
    if len(atom_name) < 4:
        atom_name_field = f' {atom_name:<3s}'
    else:
        atom_name_field = f'{atom_name:<4s}'

    return (
        f"ATOM  {serial:5d} {atom_name_field} {res_name:3s} {chain:1s}"
        f"{res_seq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"  1.00  0.00          {element:>2s}  "
    )


def write_pocket_pdb(data, out_path: Path):
    """Write a proper ATOM-record PDB from an LMDB record."""
    protein_pos  = data['protein_pos']       # [N, 3] tensor
    atom_name_id = data['protein_atom_name'] # [N] tensor (1-indexed vocab)
    aa_type      = data['protein_atom_to_aa_type']  # [N] tensor (1-indexed)
    atom2residue = data['protein_atom2residue']      # [N] tensor
    res_idx      = data['res_idx']           # [n_res] tensor (PDB residue numbers)

    if isinstance(protein_pos, torch.Tensor):
        protein_pos  = protein_pos.numpy()
        atom_name_id = atom_name_id.numpy()
        aa_type      = aa_type.numpy()
        atom2residue = atom2residue.numpy()
        res_idx      = res_idx.numpy()

    lines = []
    for i, (pos, an_id, aa_id, res_i) in enumerate(
            zip(protein_pos, atom_name_id, aa_type, atom2residue)):

        atom_name = ATOM_NAME_VOCAB.get(int(an_id), f'X{an_id}')
        res_name  = AA_VOCAB.get(int(aa_id), 'UNK')
        # Map internal residue index to PDB sequence number
        pdb_res_seq = int(res_idx[int(res_i)]) if int(res_i) < len(res_idx) else int(res_i) + 1
        element   = ELEMENT_SYMBOL.get(int(data['protein_element'][i]), 'C')

        lines.append(_pdb_atom_line(
            serial=i + 1,
            atom_name=atom_name,
            res_name=res_name,
            chain='A',
            res_seq=pdb_res_seq,
            x=float(pos[0]),
            y=float(pos[1]),
            z=float(pos[2]),
            element=element,
        ))

    lines.append('END')
    out_path.write_text('\n'.join(lines) + '\n')


def write_ligand_sdf(data, out_path: Path):
    """Write the reference ligand as a minimal SDF (atom positions only)."""
    lig_pos  = data['ligand_pos']
    lig_elem = data['ligand_element']
    if isinstance(lig_pos, torch.Tensor):
        lig_pos  = lig_pos.numpy()
        lig_elem = lig_elem.numpy()

    n = len(lig_pos)
    lines = ['\n', '     RDKit\n', '\n',
             f'{n:3d}  0  0  0  0  0  0  0  0  0999 V2000\n']
    for pos, el in zip(lig_pos, lig_elem):
        sym = ELEMENT_SYMBOL.get(int(el), 'C')
        lines.append(f'{pos[0]:10.4f}{pos[1]:10.4f}{pos[2]:10.4f} {sym:<3s} 0  0  0  0  0  0  0  0  0  0  0  0\n')
    lines.append('M  END\n$$$$\n')
    out_path.write_text(''.join(lines))


def write_docking_box(data, out_path: Path, box_size: float = 20.0):
    """Write a text file with Vina docking box parameters."""
    com = data['ligand_center_of_mass']
    if isinstance(com, torch.Tensor):
        com = com.numpy()
    cx, cy, cz = float(com[0]), float(com[1]), float(com[2])
    out_path.write_text(
        f'center_x = {cx:.3f}\n'
        f'center_y = {cy:.3f}\n'
        f'center_z = {cz:.3f}\n'
        f'size_x = {box_size:.1f}\n'
        f'size_y = {box_size:.1f}\n'
        f'size_z = {box_size:.1f}\n'
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--test_dir',  type=Path,
                   default='DiffSBDD/data/processed_expert_atomica/test')
    p.add_argument('--lmdb_path', type=Path,
                   default='data/crossdocked_pocket10_processed.lmdb')
    p.add_argument('--outdir',    type=Path, default='data/receptor_pdbs')
    p.add_argument('--n_pockets', type=int,  default=100)
    p.add_argument('--box_size',  type=float, default=20.0,
                   help='Vina docking box edge length (Å)')
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    test_files = sorted(args.test_dir.glob('complex_*.pt'))[:args.n_pockets]
    if not test_files:
        raise FileNotFoundError(f'No .pt files found in {args.test_dir}')

    print(f'Opening LMDB: {args.lmdb_path}')
    env = lmdb.open(str(args.lmdb_path), subdir=False, readonly=True,
                    lock=False, readahead=False, meminit=False)

    missing, written = 0, 0
    with env.begin() as txn:
        for pt_file in tqdm(test_files, desc='Extracting'):
            pt_data   = torch.load(pt_file, map_location='cpu')
            name_val  = pt_data['name']
            lmdb_key  = str(name_val.item() if hasattr(name_val, 'item') else name_val).encode()

            raw = txn.get(lmdb_key)
            if raw is None:
                print(f'  WARNING: key {lmdb_key} not found, skipping {pt_file.name}')
                missing += 1
                continue

            lmdb_data = pickle.loads(raw)
            pocket_name = pt_file.stem  # e.g. complex_000001

            write_pocket_pdb(lmdb_data,   args.outdir / f'{pocket_name}.pdb')
            write_ligand_sdf(lmdb_data,   args.outdir / f'{pocket_name}_ref_ligand.sdf')
            write_docking_box(lmdb_data,  args.outdir / f'{pocket_name}.box.txt',
                              box_size=args.box_size)
            written += 1

    env.close()
    print(f'\nDone. {written} pockets written to {args.outdir}/')
    if missing:
        print(f'WARNING: {missing} pockets had no matching LMDB record.')


if __name__ == '__main__':
    main()

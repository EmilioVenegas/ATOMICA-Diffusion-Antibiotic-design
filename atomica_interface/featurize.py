"""Build two-segment ATOMICA inputs from a pocket and a ligand.

ATOMICA is pretrained on intermolecular interfaces: two segments, each a sequence
of chemically-typed blocks (residues for proteins, fragments for small molecules).
`ATOMICA/data/dataset_pretrain.py` splits on `segment_ids == 0` / `== 1` and masks
each segment separately, so a single-segment input never exercises the interaction
representation at all.

This module reuses ATOMICA's own converters rather than reimplementing them, so
the inputs match what the model was trained on.
"""

import os
import sys
from typing import List, Optional

import numpy as np

# ATOMICA's modules import each other inconsistently: its own files use bare
# `from data.dataset import ...` while the vendored copies here also use
# `from ATOMICA.data.dataset import ...`. Both resolve only if the repo root AND
# the ATOMICA package directory are importable.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
for _p in (_ROOT, os.path.join(_ROOT, "ATOMICA")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ATOMICA.data.dataset import (  # noqa: E402
    Block,
    blocks_interface,
    blocks_to_data,
)
from ATOMICA.data.pdb_utils import VOCAB, Atom  # noqa: E402

# Matches "fragmentation_method" in ATOMICA/pretrain/pretrain_model_config.json.
# Using a different scheme than the checkpoint was trained with silently degrades
# the block vocabulary, so this default is tied to the pretrained weights.
DEFAULT_FRAGMENTATION = "PS_300"

# CrossDocked LMDB records encode residues as 1-based indices into the
# alphabetically-sorted three-letter codes (the `AA_NAME_NUMBER` convention used by
# the TargetDiff/Pocket2Mol preprocessing that produced `crossdocked_pocket10`).
# Verified against `residue_natoms`: every one of the 20 types matches its expected
# heavy-atom count for >97% of residues, the remainder being terminal OXT or
# incomplete side chains.
AA_NUMBER_TO_ABRV = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]

# Atomic number -> element symbol. Covers the elements CrossDocked pockets and
# ligands actually contain; anything outside it is rejected rather than guessed,
# because a wrong element silently corrupts the block chemistry.
ATOMIC_NUMBER_TO_SYMBOL = {
    1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 11: "Na", 12: "Mg",
    15: "P", 16: "S", 17: "Cl", 19: "K", 20: "Ca", 25: "Mn", 26: "Fe",
    27: "Co", 29: "Cu", 30: "Zn", 34: "Se", 35: "Br", 53: "I",
}


def pocket_blocks_from_pdb(
    pdb_path: str,
    selected_chains: Optional[List[str]] = None,
) -> List[Block]:
    """Load a protein structure as one block per residue.

    Returns a flat list across chains. Residue identity is preserved in each
    block's symbol, which is what gives ATOMICA its block-level chemistry -- the
    original pipeline discarded this by emitting a single ``UNK`` block.
    """
    from ATOMICA.data.converter.pdb_to_list_blocks import pdb_to_list_blocks

    chains = pdb_to_list_blocks(pdb_path, selected_chains=selected_chains)
    return [block for chain in chains for block in chain]


def pocket_blocks_from_arrays(
    coords,
    elements,
    atom2residue,
    residue_aa,
):
    """Build one ATOMICA block per residue from raw arrays (no PDB file needed).

    This is the LMDB counterpart of :func:`pocket_blocks_from_pdb`. The CrossDocked
    records already carry the residue grouping (``protein_atom2residue``) and the
    residue identity (``amino_acid``), so real amino-acid block symbols can be
    recovered without re-parsing a structure file.

    Returns ``(blocks, atom_index)`` where ``atom_index`` gives, for every atom
    emitted into the blocks and **in block order**, its row in the input arrays.
    Callers need this to keep any per-atom quantity aligned with the per-atom
    embeddings that come back from the encoder.

    Args:
        coords: ``[n_atoms, 3]`` atom coordinates.
        elements: ``[n_atoms]`` atomic numbers.
        atom2residue: ``[n_atoms]`` residue index for each atom.
        residue_aa: ``[n_residues]`` 1-based amino-acid codes (see
            :data:`AA_NUMBER_TO_ABRV`).
    """
    coords = np.asarray(coords, dtype=np.float64)
    elements = np.asarray(elements).astype(np.int64)
    atom2residue = np.asarray(atom2residue).astype(np.int64)
    residue_aa = np.asarray(residue_aa).astype(np.int64)

    if not (len(coords) == len(elements) == len(atom2residue)):
        raise ValueError("coords, elements and atom2residue must agree in length")

    # Stable sort groups atoms by residue while preserving intra-residue order, so
    # the emitted blocks are contiguous even if the source arrays are not.
    order = np.argsort(atom2residue, kind="stable")

    blocks: List[Block] = []
    atom_index: List[int] = []

    start = 0
    while start < len(order):
        residue = atom2residue[order[start]]
        end = start
        while end < len(order) and atom2residue[order[end]] == residue:
            end += 1

        units, kept = [], []
        for row in order[start:end]:
            symbol = ATOMIC_NUMBER_TO_SYMBOL.get(int(elements[row]))
            if symbol is None:
                # Unrecognised element: drop the atom rather than mistyping it.
                continue
            units.append(
                Atom(atom_name=symbol, coordinate=coords[row].tolist(), element=symbol)
            )
            kept.append(int(row))

        if units:
            code = int(residue_aa[residue]) if 0 <= residue < len(residue_aa) else 0
            abrv = AA_NUMBER_TO_ABRV[code - 1] if 1 <= code <= 20 else "UNK"
            blocks.append(Block(symbol=VOCAB.abrv_to_symbol(abrv), units=units))
            atom_index.extend(kept)

        start = end

    return blocks, np.asarray(atom_index, dtype=np.int64)


def ligand_blocks_from_arrays(
    coords,
    elements,
    bond_index,
    bond_type,
    fragmentation_method: Optional[str] = DEFAULT_FRAGMENTATION,
):
    """Build ATOMICA ligand blocks from raw coordinates and an explicit bond table.

    Same output as :func:`ligand_blocks_from_mol`, but sourced from the arrays a
    CrossDocked LMDB record carries rather than an RDKit molecule. Real bonds are
    required: without them the ligand cannot be fragmented into the PS_300 blocks
    the checkpoint was pretrained on.

    Args:
        coords: ``[n_atoms, 3]`` atom coordinates.
        elements: ``[n_atoms]`` atomic numbers.
        bond_index: ``[2, n_bonds]`` endpoints; both directions may be present.
        bond_type: ``[n_bonds]`` 1=single, 2=double, 3=triple, 4=aromatic, matching
            ATOMICA's ``ID2BOND``.
    """
    from ATOMICA.data.converter.atom_blocks_to_frag_blocks import (
        atom_blocks_to_frag_blocks,
    )

    coords = np.asarray(coords, dtype=np.float64)
    elements = np.asarray(elements).astype(np.int64)

    atom_blocks = []
    for row in range(len(elements)):
        symbol = ATOMIC_NUMBER_TO_SYMBOL.get(int(elements[row]))
        if symbol is None:
            raise ValueError(f"unknown ligand atomic number {int(elements[row])}")
        atom_blocks.append(
            Block(
                symbol=symbol.lower(),
                units=[
                    Atom(
                        atom_name=symbol,
                        coordinate=coords[row].tolist(),
                        element=symbol,
                        pos_code=VOCAB.atom_pos_sm,
                    )
                ],
            )
        )

    if fragmentation_method is None:
        return atom_blocks

    # The record stores each bond twice (once per direction); RDKit wants it once.
    bond_index = np.asarray(bond_index).astype(np.int64)
    bond_type = np.asarray(bond_type).astype(np.int64)
    seen, bonds = set(), []
    for k in range(bond_index.shape[1]):
        src, dst = int(bond_index[0, k]), int(bond_index[1, k])
        if src == dst:
            continue
        key = (min(src, dst), max(src, dst))
        if key in seen:
            continue
        seen.add(key)
        bonds.append((key[0], key[1], int(bond_type[k])))

    return atom_blocks_to_frag_blocks(
        atom_blocks, bonds=bonds, fragmentation_method=fragmentation_method
    )


def ligand_blocks_from_mol(mol, fragmentation_method: Optional[str] = DEFAULT_FRAGMENTATION):
    """Convert an RDKit molecule with a 3D conformer into ATOMICA blocks.

    Mirrors `ATOMICA/data/converter/sm_pdb_to_blocks.py`, but takes a molecule
    object so conformers generated from SMILES can be used directly rather than
    round-tripping through a file.

    With ``fragmentation_method`` set, atoms are grouped into the principal-subgraph
    fragments the checkpoint was pretrained on; with ``None`` each atom becomes its
    own block.
    """
    from rdkit.Chem.rdchem import GetPeriodicTable

    from ATOMICA.data.converter.atom_blocks_to_frag_blocks import (
        atom_blocks_to_frag_blocks,
    )

    if mol.GetNumConformers() == 0:
        raise ValueError("ligand molecule has no 3D conformer")

    periodic_table = GetPeriodicTable()
    coords = mol.GetConformer().GetPositions()

    blocks = []
    for atom, coord in zip(mol.GetAtoms(), coords):
        element = periodic_table.GetElementSymbol(atom.GetAtomicNum())
        unit = Atom(
            atom_name=element,
            coordinate=coord.tolist(),
            element=element,
            pos_code=VOCAB.atom_pos_sm,
        )
        blocks.append(Block(symbol=element.lower(), units=[unit]))

    if fragmentation_method is None:
        return blocks

    bonds = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), int(b.GetBondTypeAsDouble()))
        for b in mol.GetBonds()
    ]
    return atom_blocks_to_frag_blocks(
        blocks, bonds=bonds, fragmentation_method=fragmentation_method
    )


def component_smiles(code: str) -> Optional[str]:
    """Look up a PDB chemical-component SMILES from ATOMICA's bundled table.

    The table is ``SMILES<TAB>CODE<TAB>name``. Needed because a ligand read from a
    PDB has coordinates but no reliable bond orders, and correct bonds are a
    prerequisite for fragmenting it the way the checkpoint expects.
    """
    table = os.path.join(
        _ROOT, "ATOMICA", "data", "converter", "pdb_chemical_components_smiles.txt"
    )
    code = code.strip().upper()
    with open(table) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1].strip().upper() == code:
                return parts[0]
    return None


def ligand_from_pdb_het(pdb_path: str, resname: str, chain: Optional[str] = None,
                        smiles: Optional[str] = None):
    """Extract a HETATM ligand from a PDB as an RDKit molecule with real bonds.

    PDB records carry no bond orders, so they are assigned from the component's
    SMILES template. Without that the molecule cannot be fragmented into the
    PS_300 blocks the pretrained model was trained on, and we would be back to
    feeding ATOMICA an input it has never seen.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    keep = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("HETATM") or line[17:20].strip() != resname:
                continue
            if chain and line[21] != chain:
                continue
            keep.append(line)
    if not keep:
        raise ValueError(f"no HETATM {resname!r} (chain {chain}) in {pdb_path}")

    mol = Chem.MolFromPDBBlock("".join(keep) + "END\n", removeHs=True, sanitize=False)
    if mol is None:
        raise ValueError(f"RDKit could not parse {resname} from {pdb_path}")

    smiles = smiles or component_smiles(resname)
    if smiles is None:
        raise ValueError(
            f"no SMILES template for {resname}; pass smiles= explicitly, otherwise "
            "bond orders are unknown and fragmentation would be wrong"
        )
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise ValueError(f"could not parse template SMILES for {resname}: {smiles}")

    return AllChem.AssignBondOrdersFromTemplate(template, mol)


def interface_data(pocket_blocks, ligand_blocks, dist_th: float = 8.0, trim: bool = True):
    """Assemble a two-segment ATOMICA record: pocket = segment 0, ligand = segment 1.

    With ``trim`` (the default) only blocks within ``dist_th`` of the other entity
    are kept, matching how ATOMICA's own pipeline restricts a complex to its
    interface. Disable it to encode the entities in full.
    """
    if not pocket_blocks:
        raise ValueError("no pocket blocks")
    if not ligand_blocks:
        raise ValueError("no ligand blocks")

    if trim:
        pocket_blocks, ligand_blocks = blocks_interface(
            pocket_blocks, ligand_blocks, dist_th
        )
        if not pocket_blocks or not ligand_blocks:
            raise ValueError(
                f"no contacts within {dist_th} A -- ligand is not placed in the pocket"
            )

    # Segment ids are assigned by argument order: pocket 0, ligand 1.
    return blocks_to_data(pocket_blocks, ligand_blocks)


def to_batch(data, device="cpu"):
    """Convert a `blocks_to_data` record into the tensor batch `infer()` expects.

    `infer()` consumes X, B, A, block_lengths, lengths and segment_ids;
    `blocks_to_data` supplies everything except ``lengths``, which is the block
    count per complex used to split a batch.
    """
    import torch

    block_lengths = torch.tensor(data["block_lengths"], dtype=torch.long, device=device)
    return {
        "X": torch.tensor(np.asarray(data["X"], dtype=np.float32), device=device),
        "B": torch.tensor(data["B"], dtype=torch.long, device=device),
        "A": torch.tensor(data["A"], dtype=torch.long, device=device),
        "block_lengths": block_lengths,
        "lengths": torch.tensor([len(block_lengths)], dtype=torch.long, device=device),
        "segment_ids": torch.tensor(data["segment_ids"], dtype=torch.long, device=device),
    }


def atom_segment_ids(data) -> np.ndarray:
    """Expand a record's per-block segment ids to one entry per atom.

    ``infer()`` returns ``unit_repr`` with one row per atom in the same order as
    ``data['X']``, but segment membership is stored per block. Expanding it by
    ``block_lengths`` is what lets a caller pick out just the pocket's rows and keep
    them aligned with the pocket's coordinates.
    """
    block_lengths = np.asarray(data["block_lengths"], dtype=np.int64)
    segment_ids = np.asarray(data["segment_ids"], dtype=np.int64)
    return np.repeat(segment_ids, block_lengths)


def summarize(data) -> dict:
    """Report the block/segment structure of a record.

    Used by the tests and the Phase 0 driver to assert that a record really has
    two populated segments with varied block types, which is exactly the property
    the original featurization lacked.
    """
    segment_ids = np.asarray(data["segment_ids"])
    block_types = np.asarray(data["B"])
    return {
        "n_atoms": len(data["A"]),
        "n_blocks": len(block_types),
        "n_segments": int(len(np.unique(segment_ids))),
        "blocks_per_segment": {
            int(s): int((segment_ids == s).sum()) for s in np.unique(segment_ids)
        },
        "distinct_block_types": int(len(np.unique(block_types))),
    }

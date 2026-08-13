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

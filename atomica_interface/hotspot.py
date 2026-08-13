"""Interaction hotspot fields: score chemical probes on a grid through a pocket.

Places small probes (apolar, donor, acceptor, aromatic, charged) at grid points
in a binding site and scores each placement as a genuine two-segment ATOMICA
interface, giving a field over space x probe type: which subpocket favours which
interaction chemistry.

This is a **within-pocket** task by construction, which matters. Absolute ATOMICA
scores do not transfer across systems (`results/pose_scorer/README.md`), so
fields are converted to percentile ranks inside each pocket-and-probe field and
never compared as raw values between targets. That is also the convention used by
Fragment Hotspot Maps, whose validation protocol we adopt.

Design decisions taken from the prior-art survey:

* **The pocket is fixed across every grid point** (`trim=False`). Re-trimming per
  probe position would let the field vary with how many residues each placement
  happens to contact -- the composition shortcut measured at AUROC ~0.70 in
  Phase 0.
* **Clashing and solvent-inaccessible points are masked out before scoring.**
  A rigid grid is otherwise dominated by "is this point too close to protein":
  minimum contact distance alone reached AUROC 1.000 on uncontrolled Phase 0
  poses. Buried but unreachable cavities score well for probes no ligand could
  put there, which is why FTMap requires probe solvent accessibility.
* **Each probe field is normalised independently.** Benzene has six heavy atoms
  and water has one, so any size-dependent score would rank benzene above water
  everywhere. Percentile rank within a field removes this for free.

Known limitation to check for, not to fix here: ATOMICA never saw explicit water,
so polar probes are expected to be over-rewarded in buried polar pockets.
"""

from typing import Dict, List, Optional

import numpy as np

# Probe SMILES chosen to cover the five interaction types Fragment Hotspot Maps
# uses, so the per-atom type mapping in validation is one-to-one.
DEFAULT_PROBES: Dict[str, str] = {
    "apolar": "C",            # methane
    "aromatic": "c1ccccc1",   # benzene
    "donor": "CO",            # methanol, donates through its hydroxyl
    "acceptor": "CC=O",       # acetaldehyde carbonyl
    "positive": "C[NH3+]",    # methylammonium
    "negative": "CC(=O)[O-]",  # acetate
}

# Ligand heavy atoms are assigned to the probe whose chemistry they match, so a
# ligand atom is looked up in the corresponding probe's field.
def classify_ligand_atom(atom) -> str:
    """Map an RDKit heavy atom onto one of the probe types."""
    symbol = atom.GetSymbol()
    charge = atom.GetFormalCharge()
    if charge > 0:
        return "positive"
    if charge < 0:
        return "negative"
    if atom.GetIsAromatic():
        return "aromatic"
    if symbol == "C":
        return "apolar"
    if symbol in ("N", "O"):
        # An N/O carrying hydrogen can donate; otherwise treat it as an acceptor.
        return "donor" if atom.GetTotalNumHs() > 0 else "acceptor"
    if symbol == "S":
        return "apolar"
    return "apolar"


def build_probe(smiles: str, seed: int = 0):
    """Embed a probe SMILES as a single 3D conformer, hydrogens removed."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"bad probe SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise ValueError(f"could not embed probe: {smiles}")
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    return Chem.RemoveHs(mol)


def pocket_grid(
    site_coords: np.ndarray,
    spacing: float = 1.0,
    padding: float = 2.0,
    min_contact: float = 3.0,
    max_contact: float = 5.5,
) -> np.ndarray:
    """Grid points inside the site that a probe atom could plausibly occupy.

    Two filters, both required:

    * ``min_contact`` drops points clashing into protein atoms. Without it the
      field mostly reports steric overlap.
    * ``max_contact`` drops points out in bulk solvent, which are trivially
      non-clashing and would dominate a "favourable" field with empty space.

    The surviving shell is the region a ligand atom could actually sit in.
    """
    lo = site_coords.min(axis=0) - padding
    hi = site_coords.max(axis=0) + padding
    axes = [np.arange(lo[i], hi[i] + spacing, spacing) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    # Chunked so a fine grid over a large site does not allocate an
    # (n_grid x n_atoms) distance matrix all at once.
    keep = np.zeros(len(grid), dtype=bool)
    for start in range(0, len(grid), 20000):
        chunk = grid[start : start + 20000]
        d = np.linalg.norm(chunk[:, None, :] - site_coords[None, :, :], axis=2).min(1)
        keep[start : start + 20000] = (d >= min_contact) & (d <= max_contact)
    return grid[keep]


def buriedness(points: np.ndarray, site_coords: np.ndarray, radius: float = 8.0) -> np.ndarray:
    """Protein heavy atoms within ``radius`` of each point.

    This is the confound baseline, not a feature. Grid scores correlate with
    enclosure in every method in this literature -- Fragment Hotspot Maps has to
    weight its propensity maps by buriedness precisely because otherwise large
    regions of the protein score highly. Any hotspot claim must be shown to beat
    this, and to survive controlling for it.
    """
    counts = np.zeros(len(points), dtype=np.float32)
    for start in range(0, len(points), 20000):
        chunk = points[start : start + 20000]
        d = np.linalg.norm(chunk[:, None, :] - site_coords[None, :, :], axis=2)
        counts[start : start + 20000] = (d <= radius).sum(axis=1)
    return counts


def place(mol, centre: np.ndarray, rotation: Optional[np.ndarray] = None):
    """Copy a probe with its centroid at ``centre``, optionally rotated."""
    from copy import deepcopy

    out = deepcopy(mol)
    conf = out.GetConformer()
    coords = conf.GetPositions()
    local = coords - coords.mean(axis=0)
    if rotation is not None:
        local = local @ rotation.T
    for i, xyz in enumerate(local + centre):
        conf.SetAtomPosition(i, [float(v) for v in xyz])
    return out


def random_rotations(n: int, rng) -> List[np.ndarray]:
    """``n`` uniformly random proper rotations.

    Anisotropic probes (benzene especially) score very differently by
    orientation, so a rigid single-orientation grid understates them. Taking the
    best of several orientations is a cheap stand-in for the local minimisation
    MCSS performs.
    """
    out = []
    for _ in range(n):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q *= np.sign(np.diag(r))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        out.append(q)
    return out


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Convert scores to within-field percentile ranks (higher = more favourable).

    Raw ATOMICA scores are not comparable across pockets or across probe types,
    so every comparison in this module is made on ranks computed inside a single
    pocket-and-probe field.
    """
    order = np.argsort(np.argsort(values))
    return 100.0 * order / max(len(values) - 1, 1)

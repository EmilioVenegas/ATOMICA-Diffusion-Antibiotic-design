from pathlib import Path

# 3-letter → 1-letter amino acid mapping
AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLU": "E", "GLN": "Q", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def get_protein_sequence_from_pdb(
    pdb_path: str | Path,
    chain_id: str | None = None,
) -> str:
    """
    Extract protein sequence from a PDB file.

    Parameters
    ----------
    pdb_path : str or Path
        Path to PDB file.
    chain_id : str or None
        If provided, extract only this chain. Otherwise use first found.

    Returns
    -------
    str
        Protein sequence (1-letter amino acids).
    """
    pdb_path = Path(pdb_path)

    # 1. Try SEQRES (best)
    seqres = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("SEQRES"):
                chain = line[11].strip()
                if chain_id and chain != chain_id:
                    continue
                residues = line[19:].split()
                seqres.setdefault(chain, []).extend(residues)

    if seqres:
        # pick requested chain or the first one
        chain = chain_id or next(iter(seqres))
        return "".join(AA3_TO_AA1.get(res, "X") for res in seqres[chain])

    # 2. Fallback: reconstruct from ATOM records
    residues = []
    seen = set()

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            resname = line[17:20].strip()
            chain = line[21].strip()
            resseq = line[22:26].strip()

            if chain_id and chain != chain_id:
                continue

            key = (chain, resseq)
            if key in seen:
                continue

            seen.add(key)
            residues.append(AA3_TO_AA1.get(resname, "X"))

    if not residues:
        raise ValueError(f"No protein residues found in {pdb_path}")

    return "".join(residues)

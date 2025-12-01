import argparse

from rdkit import Chem
from rdkit.Chem import AllChem

# 20 reasonably drug-like example SMILES (aspirin, ibuprofen, caffeine, etc.)
EXAMPLE_LIGANDS = [
    ("aspirin",        "CC(=O)OC1=CC=CC=C1C(=O)O"),
    ("caffeine",       "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
    ("ibuprofen",      "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"),
    ("acetaminophen",  "CC(=O)NC1=CC=C(C=C1)O"),
    ("naproxen",       "COC1=CC=CC2=C1C=C(C=C2)CC(C)C(=O)O"),
    ("fluconazole",    "OC(Cn1cnc(c1F)F)(Cn2cnc(c2F)F)CO"),
    ("metformin",      "CNC(=N)NC(=N)N"),
    ("amoxicillin",    "CC1(C)SCC(N1C(=O)NCC2=CC=CC=C2)C(=O)O"),
    ("clavulanic",     "CC1(C(=O)N2C1SC(C2=O)O)C(=O)O"),
    ("levodopa",       "C1=CC(=C(C=C1C(C(C(=O)O)N)O)O)O"),
    ("atorvastatin",   "CC(C)C1=CC(=O)N(C(=O)N1C)CC(C(C2=CC=CC=C2)O)O"),
    ("rosuvastatin",   "CC(C)CC(C(=O)O)C(C(C1=CC=CC=C1)O)O"),
    ("omeprazole",     "COC1=CC=C(C=C1)C2=NC(=NO2)C3=CC=CC=C3O"),
    ("lisinopril",     "CC(C)C(C(=O)N1CCCC1C(=O)O)NCC(=O)O"),
    ("losartan",       "CC1=NC(=NC(=N1)NC2=CC=CC=C2)C3=CC=CC=C3Cl"),
    ("simvastatin",    "CCC(C)C1CCC2(C(C1)CCC(C2(C)C(=O)OCC3=CC=CC=C3)O)C"),
    ("pravastatin",    "CC(C)CC(C(=O)O)C(C(C1=CC=CC=C1)O)O"),
    ("indomethacin",   "CC(C1=CC2=C(C=C1)NC(=O)C2=O)C(=O)O"),
    ("ranitidine",     "CN(C)CCNC(=S)NC1=NC(=CC=C1)N"),
    ("sildenafil",     "CCN(CC)CCOC1=CC2=NC(=O)N(C2=NC=N1)C3=CC=CC=C3"),
]


def make_3d_conformer(smiles, name, sample_idx):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, randomSeed=sample_idx + 1)
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        # Fall back to 2D if 3D fails
        pass
    
    mol.SetProp("_Name", name)
    mol.SetProp("sample_idx", str(sample_idx))
    mol.SetProp("smiles", smiles)
    
    return mol


def main():
    parser = argparse.ArgumentParser(
        description="Mock ligand generator for RL loop (no DiffSBDD checkpoint needed)."
    )
    parser.add_argument(
        "--outfile",
        type=str,
        required=True,
        help="Output SDF file (this will mimic generate_ligands.py output).",
    )
    args = parser.parse_args()
    
    writer = Chem.SDWriter(args.outfile)
    for i, (name, smi) in enumerate(EXAMPLE_LIGANDS):
        mol = make_3d_conformer(smi, name, i)
        if mol is not None:
            writer.write(mol)
    
    writer.close()
    print(f"Wrote {len(EXAMPLE_LIGANDS)} mock ligands to {args.outfile}")


if __name__ == "__main__":
    main()


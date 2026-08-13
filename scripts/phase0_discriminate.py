"""Phase 0 go/no-go: does ATOMICA's interface representation encode binding?

For one pocket, embed (pocket, ligand) complexes where the ligand is a known
active, a property-matched decoy, or a random drug-like molecule, and test whether
a cross-validated linear probe can separate actives from the others.

Three feature sets are compared, and the comparison is the point:

  interface   two-segment (pocket + ligand) ATOMICA representation
  ligand_only single-segment ATOMICA representation of the ligand alone
  physchem    MW / logP / TPSA / HBD / HBA / rotatable bonds

The controls decide what a positive result means:

  * interface ~ ligand_only  -> the interaction modelling adds nothing; ATOMICA is
    acting as a molecular encoder, and conditioning a generator on pocket
    embeddings cannot recover pocket specificity.
  * interface ~ physchem     -> ATOMICA is recapitulating drug-likeness, not
    binding. This is the same failure mode the A/B ablation hit, diagnosed cheaply.
  * interface >> both        -> the interaction representation carries binding
    signal, and Phases 1-4 of docs/experiment-plan.md are justified.

Scores are out-of-fold, so no sample is ranked by a model that saw it, and a label
permutation test guards against small-sample optimism.

Usage (from repo root):

    python scripts/phase0_discriminate.py \
        --pocket data/cdk2.pdb \
        --actives scripts/eval/cdk2_test_data/binders.csv \
        --decoys scripts/eval/cdk2_test_data/decoys.csv \
        --random scripts/eval/cdk2_test_data/random_molecules.csv \
        --atomica_config ATOMICA/pretrain/pretrain_model_config.json \
        --atomica_weights ATOMICA/pretrain/pretrain_model_weights.pt \
        --out results/phase0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from atomica_interface.scoring import (  # noqa: E402
    cross_validated_probe,
    enrichment_factor,
    permutation_test,
    roc_auc,
)

DESCRIPTOR_NAMES = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds"]


def read_smiles(path):
    """Read a CSV with a `smiles` column, returning (smiles, id) pairs."""
    import csv

    rows = []
    with open(path, newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            key = next((k for k in row if k and k.lower() == "smiles"), None)
            if key is None:
                raise ValueError(f"{path} has no 'smiles' column")
            name = row.get("id") or row.get("name") or f"{Path(path).stem}_{i}"
            rows.append((row[key], name))
    return rows


def embed_ligand(smiles, seed=0):
    """SMILES -> a single 3D conformer, MMFF-optimised where possible."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass  # unparameterised atoms: keep the unrefined ETKDG conformer
    return Chem.RemoveHs(mol)


def reference_ligand_centre(pdb_path, resname, chain=None):
    """Centroid of a HETATM residue, used as the binding-site centre.

    The whole-protein centroid is not the binding site and must not be used as a
    placement target: in 1H1S the 4SP inhibitor sits ~13.5 A from chain A's
    centroid, which would place every candidate outside the pocket entirely.
    """
    coords = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("HETATM") or line[17:20].strip() != resname:
                continue
            if chain and line[21] != chain:
                continue
            coords.append([float(line[30 + 8 * i : 38 + 8 * i]) for i in range(3)])
    if not coords:
        raise ValueError(f"no HETATM residue {resname!r} (chain {chain}) in {pdb_path}")
    return np.asarray(coords, dtype=float).mean(axis=0)


def place_in_pocket(mol, centre):
    """Translate a conformer so its centroid sits at ``centre``.

    A conformer generated from SMILES has no pose relative to the receptor.
    Centring gives every candidate the same, ligand-independent placement, which
    keeps the comparison fair -- but it fixes only position, not orientation or
    conformation. This is a stand-in for docking, so a *null* result under this
    placement is not conclusive about ATOMICA. Supply real poses with
    --poses_sdf for a fair test.
    """
    conf = mol.GetConformer()
    coords = conf.GetPositions()
    shifted = coords - coords.mean(axis=0) + centre
    for i, xyz in enumerate(shifted):
        conf.SetAtomPosition(i, xyz.tolist())
    return mol


def physchem_features(mol):
    from rdkit.Chem import Descriptors

    return [getattr(Descriptors, name)(mol) for name in DESCRIPTOR_NAMES]


def evaluate(features, labels, name, seed=0, n_permutations=2000):
    """Cross-validated probe -> AUROC, EF@5%, permutation p-value."""
    features = np.asarray(features, dtype=float)
    scores = cross_validated_probe(features, labels, seed=seed)
    if scores is None:
        return {"feature_set": name, "error": "too few samples per class to stratify"}
    return {
        "feature_set": name,
        "n": int(len(labels)),
        "n_positive": int(np.sum(labels)),
        "dim": int(features.shape[1]),
        "auroc": round(roc_auc(labels, scores), 4),
        "ef_at_5pct": round(enrichment_factor(labels, scores, 0.05), 3),
        "permutation_p": round(permutation_test(labels, scores, n_permutations, seed), 5),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pocket", required=True, help="receptor PDB")
    p.add_argument("--chains", nargs="*", default=None)
    p.add_argument(
        "--ref_ligand",
        default=None,
        help="HETATM residue name of the co-crystal ligand (e.g. 4SP). Its centroid "
        "defines the binding site. Without it the whole-protein centroid is used, "
        "which is NOT the pocket.",
    )
    p.add_argument("--ref_ligand_chain", default=None)
    p.add_argument("--actives", required=True)
    p.add_argument("--decoys", required=True)
    p.add_argument("--random", dest="random_mols", default=None)
    p.add_argument("--atomica_config", default="ATOMICA/pretrain/pretrain_model_config.json")
    p.add_argument("--atomica_weights", default="ATOMICA/pretrain/pretrain_model_weights.pt")
    p.add_argument("--dist_th", type=float, default=8.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/phase0")
    args = p.parse_args()

    import torch

    from atomica_interface.featurize import (
        interface_data,
        ligand_blocks_from_mol,
        pocket_blocks_from_pdb,
        summarize,
        to_batch,
    )
    from atomica_interface.scoring import interface_representation, load_encoder

    device = args.device if torch.cuda.is_available() else "cpu"
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] loading pocket {args.pocket}")
    pocket_blocks = pocket_blocks_from_pdb(args.pocket, args.chains)
    protein_centre = np.mean(
        [atom.get_coord() for block in pocket_blocks for atom in block.units], axis=0
    )
    if args.ref_ligand:
        pocket_centre = reference_ligand_centre(
            args.pocket, args.ref_ligand, args.ref_ligand_chain
        )
        offset = float(np.linalg.norm(pocket_centre - protein_centre))
        print(
            f"      {len(pocket_blocks)} residue blocks; site centre from {args.ref_ligand} "
            f"at {np.round(pocket_centre, 1)} ({offset:.1f} A from the protein centroid)"
        )
    else:
        pocket_centre = protein_centre
        print(
            f"      {len(pocket_blocks)} residue blocks; WARNING: no --ref_ligand, "
            f"falling back to the protein centroid {np.round(pocket_centre, 1)}, "
            "which is generally not the binding site"
        )

    print(f"[2/4] loading ATOMICA encoder on {device}")
    model = load_encoder(args.atomica_config, args.atomica_weights, device)

    groups = [(args.actives, 1, "active"), (args.decoys, 0, "decoy")]
    if args.random_mols:
        groups.append((args.random_mols, 0, "random"))

    interface_feats, ligand_feats, physchem, labels, kinds, names = [], [], [], [], [], []
    skipped = []
    structure_log = None

    print("[3/4] embedding ligands and encoding complexes")
    for path, label, kind in groups:
        for smiles, name in read_smiles(path):
            mol = embed_ligand(smiles, seed=args.seed)
            if mol is None:
                skipped.append({"name": name, "reason": "conformer generation failed"})
                continue
            mol = place_in_pocket(mol, pocket_centre)

            try:
                lig_blocks = ligand_blocks_from_mol(mol)
                data = interface_data(pocket_blocks, lig_blocks, args.dist_th)
                if structure_log is None:
                    structure_log = summarize(data)
                    print(f"      example record: {structure_log}")
                interface_feats.append(
                    interface_representation(model, to_batch(data, device)).reshape(-1)
                )
                # Control: same ligand, no pocket -> single segment.
                lig_only = interface_data(lig_blocks, lig_blocks, args.dist_th, trim=False)
                ligand_feats.append(
                    interface_representation(model, to_batch(lig_only, device)).reshape(-1)
                )
            except Exception as exc:  # noqa: BLE001
                skipped.append({"name": name, "reason": str(exc)})
                continue

            physchem.append(physchem_features(mol))
            labels.append(label)
            kinds.append(kind)
            names.append(name)

    labels = np.asarray(labels)
    if len(labels) == 0 or labels.sum() == 0:
        raise SystemExit("no usable molecules -- check inputs")
    print(f"      encoded {len(labels)} molecules ({labels.sum()} active); skipped {len(skipped)}")

    print("[4/4] cross-validated probes")
    results = [
        evaluate(interface_feats, labels, "interface", args.seed),
        evaluate(ligand_feats, labels, "ligand_only", args.seed),
        evaluate(physchem, labels, "physchem", args.seed),
    ]

    report = {
        "pocket": args.pocket,
        "n_molecules": int(len(labels)),
        "counts_by_kind": {k: kinds.count(k) for k in set(kinds)},
        "record_structure": structure_log,
        "results": results,
        "skipped": skipped,
        "caveats": [
            "Ligand poses are centroid-placed conformers, not docked. A null result "
            "may reflect pose quality rather than the representation.",
            "Small n: treat this as a smoke test and confirm on DUD-E or LIT-PCBA "
            "before acting on the outcome.",
        ],
    }
    (outdir / "phase0_report.json").write_text(json.dumps(report, indent=2))

    print(f"\n{'feature set':<14}{'AUROC':>8}{'EF@5%':>8}{'perm p':>10}")
    for r in results:
        if "error" in r:
            print(f"{r['feature_set']:<14}{r['error']}")
        else:
            print(f"{r['feature_set']:<14}{r['auroc']:>8.3f}{r['ef_at_5pct']:>8.2f}{r['permutation_p']:>10.4f}")

    interface = next((r for r in results if r["feature_set"] == "interface"), {})
    auroc = interface.get("auroc")
    if auroc is not None:
        best_control = max(
            (r.get("auroc", 0) for r in results if r["feature_set"] != "interface"), default=0
        )
        print()
        if auroc < 0.6:
            print("VERDICT: no usable signal. Do not build conditioning on this.")
        elif auroc <= best_control + 0.05:
            print(
                "VERDICT: interface does not beat its controls -- ATOMICA is acting as a "
                "molecule/property encoder here, not an interaction model."
            )
        else:
            print("VERDICT: interface representation carries signal beyond both controls.")
    print(f"\nwrote {outdir/'phase0_report.json'}")


if __name__ == "__main__":
    main()

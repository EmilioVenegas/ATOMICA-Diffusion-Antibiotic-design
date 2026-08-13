"""Build a CASF-style pose benchmark from open RCSB data.

CASF-2016 is the standard benchmark for docking power but sits behind
registration. This constructs an equivalent set from freely available structures:
select protein-ligand complexes, redock each ligand into its own pocket to
generate decoy poses, and label every pose by RMSD to the crystal pose.

A scorer has "docking power" if it ranks the near-native pose (RMSD <= 2 A) top.
Because decoys come from an actual docking engine they are physically plausible
and clash-free, unlike the rigid perturbations used in the Phase 0 gate -- which
were separable on steric overlap alone (see results/phase0/README.md).

The manifest this writes is the same shape CASF-2016 would produce, so the
training and evaluation scripts consume either.

Usage (from repo root):

    python scripts/build_pose_benchmark.py --n_targets 30 --out data/pose_benchmark
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

# Crystallisation additives, buffers, ions and cryoprotectants. Heavy-atom count
# filters most of these out anyway; this catches the drug-sized ones.
NON_LIGANDS = {
    "HOH", "DOD", "SO4", "PO4", "GOL", "EDO", "PEG", "PGE", "1PE", "MES", "TRS",
    "ACT", "ACY", "FMT", "DMS", "IPA", "MPD", "NO3", "CIT", "TAR", "MLI", "EPE",
    "BME", "CME", "IMD", "BTB", "MRD", "P6G", "2PE", "SIN", "SCN", "AZI", "FLC",
    "NA", "CL", "MG", "ZN", "CA", "K", "MN", "FE", "FE2", "NI", "CU", "CD", "CO",
    "HG", "BR", "IOD", "F", "CS", "RB", "SR", "BA", "PB", "PT", "AU", "AG",
}


def rcsb_candidates(limit):
    """Entry IDs for single-protein X-ray structures with a ligand of interest."""
    query = {
        "query": {"type": "group", "logical_operator": "and", "nodes": [
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_entry_info.experimental_method",
                "operator": "exact_match", "value": "X-ray"}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "less_or_equal", "value": 2.2}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                "operator": "equals", "value": 1}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_nonpolymer_entity_annotation.type",
                "operator": "exact_match", "value": "SUBJECT_OF_INVESTIGATION"}},
        ]},
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": limit},
            "results_content_type": ["experimental"],
            # Sorted for determinism, then shuffled with a fixed seed by the
            # caller: consecutive PDB IDs are often the same protein from one
            # deposition, which would make the "targets" far from independent.
            "sort": [{"sort_by": "rcsb_accession_info.initial_release_date",
                      "direction": "desc"}],
        },
    }
    url = ("https://search.rcsb.org/rcsbsearch/v2/query?json="
           + urllib.parse.quote(json.dumps(query)))
    with urllib.request.urlopen(url, timeout=90) as fh:
        return [x["identifier"] for x in json.load(fh)["result_set"]]


def fetch_pdb(pdb_id, dest):
    if dest.exists():
        return dest
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        with urllib.request.urlopen(url, timeout=90) as r:
            dest.write_bytes(r.read())
    except Exception:
        return None
    return dest


def choose_ligand(pdb_path, min_atoms, max_atoms):
    """Pick the largest drug-sized HETATM residue that has a SMILES template.

    Bond orders are unknown in a PDB, so a template is mandatory: without it the
    ligand cannot be fragmented with the PS_300 scheme the checkpoint expects.
    """
    from atomica_interface.featurize import component_smiles

    counts = {}
    for line in pdb_path.read_text().splitlines():
        if not line.startswith("HETATM"):
            continue
        resname = line[17:20].strip()
        if resname in NON_LIGANDS or line[76:78].strip() == "H":
            continue
        key = (resname, line[21], line[22:27].strip())
        counts[key] = counts.get(key, 0) + 1

    for (resname, chain, _), n in sorted(counts.items(), key=lambda kv: -kv[1]):
        if not (min_atoms <= n <= max_atoms):
            continue
        smiles = component_smiles(resname)
        if smiles:
            return resname, chain, n, smiles
    return None


def write_receptor(pdb_path, chain, dest):
    """Protein-only receptor: ATOM records for one chain, no waters or ligands."""
    lines = [
        ln for ln in pdb_path.read_text().splitlines()
        if ln.startswith("ATOM") and ln[21] == chain
    ]
    if not lines:
        return None
    dest.write_text("\n".join(lines) + "\nEND\n")
    return dest


def redock(receptor, native_sdf, out_sdf, num_modes, exhaustiveness, seed,
           autobox_add=6, min_rmsd_filter=0.75):
    cmd = [
        "smina", "-r", str(receptor), "-l", str(native_sdf),
        "--autobox_ligand", str(native_sdf), "--autobox_add", str(autobox_add),
        "--num_modes", str(num_modes), "--exhaustiveness", str(exhaustiveness),
        "--min_rmsd_filter", str(min_rmsd_filter),
        "--seed", str(seed), "-o", str(out_sdf), "--cpu", "4",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    return proc.returncode == 0 and out_sdf.exists()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_targets", type=int, default=30)
    p.add_argument("--candidate_pool", type=int, default=400)
    p.add_argument("--min_atoms", type=int, default=16)
    p.add_argument("--max_atoms", type=int, default=45)
    p.add_argument("--num_modes", type=int, default=20)
    p.add_argument("--autobox_add", type=float, default=6.0,
                   help="box padding (A). Larger gives poses that land off-site, "
                        "which is what makes the ranking task non-trivial.")
    p.add_argument("--min_rmsd_filter", type=float, default=0.75,
                   help="smina pose diversity filter")
    p.add_argument("--exhaustiveness", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/pose_benchmark")
    args = p.parse_args()

    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, rdMolAlign

    RDLogger.DisableLog("rdApp.*")
    from atomica_interface.featurize import ligand_from_pdb_het

    out = Path(args.out)
    (out / "structures").mkdir(parents=True, exist_ok=True)
    (out / "poses").mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.csv"

    done = set()
    if manifest_path.exists():  # resumable
        with open(manifest_path) as fh:
            done = {r["target"] for r in csv.DictReader(fh)}
        print(f"resuming: {len(done)} targets already built")

    rows = []
    accepted = len(done)
    print(f"querying RCSB for up to {args.candidate_pool} candidates ...")
    import random

    candidates = rcsb_candidates(args.candidate_pool)
    random.Random(args.seed).shuffle(candidates)
    for pdb_id in candidates:
        if accepted >= args.n_targets:
            break
        if pdb_id in done:
            continue

        pdb_path = fetch_pdb(pdb_id, out / "structures" / f"{pdb_id}.pdb")
        if pdb_path is None:
            continue
        pick = choose_ligand(pdb_path, args.min_atoms, args.max_atoms)
        if pick is None:
            continue
        resname, chain, n_atoms, _ = pick

        try:
            native = ligand_from_pdb_het(str(pdb_path), resname, chain)
        except Exception:
            continue

        receptor = write_receptor(pdb_path, chain, out / "structures" / f"{pdb_id}_rec.pdb")
        if receptor is None:
            continue

        native_sdf = out / "poses" / f"{pdb_id}_native.sdf"
        w = Chem.SDWriter(str(native_sdf))
        w.write(native)
        w.close()

        poses_sdf = out / "poses" / f"{pdb_id}_docked.sdf"
        try:
            ok = redock(receptor, native_sdf, poses_sdf, args.num_modes,
                        args.exhaustiveness, args.seed, args.autobox_add,
                        args.min_rmsd_filter)
        except subprocess.TimeoutExpired:
            ok = False
        if not ok:
            continue

        template = Chem.MolFromSmiles(pick[3])
        if template is None:
            continue
        supplier = Chem.SDMolSupplier(str(poses_sdf), sanitize=False)
        n_kept = 0
        for i, pose in enumerate(supplier):
            if pose is None:
                continue
            try:
                # smina writes polar hydrogens and no bond orders, while the
                # native carries template bond orders and heavy atoms only.
                # Both have to be reconciled or the substructure match that
                # GetBestRMS needs cannot succeed.
                heavy = Chem.RemoveAllHs(pose, sanitize=False)
                fixed = AllChem.AssignBondOrdersFromTemplate(template, heavy)
                # Symmetry-aware: equivalent atoms are matched rather than
                # compared by index, so a flipped phenyl is not counted as error.
                rmsd = rdMolAlign.GetBestRMS(fixed, native)
            except Exception:
                continue
            score = pose.GetProp("minimizedAffinity") if pose.HasProp("minimizedAffinity") else ""
            rows.append({
                "target": pdb_id, "ligand": resname, "chain": chain,
                "pose_index": i, "rmsd": round(float(rmsd), 3),
                "smina_score": score, "n_ligand_atoms": n_atoms,
                "poses_file": str(poses_sdf), "receptor": str(receptor),
                "native_sdf": str(native_sdf),
            })
            n_kept += 1

        if n_kept == 0:
            continue
        accepted += 1
        best = min(r["rmsd"] for r in rows if r["target"] == pdb_id)
        print(f"  [{accepted}/{args.n_targets}] {pdb_id} {resname} "
              f"({n_atoms} atoms): {n_kept} poses, best RMSD {best:.2f} A")

        header = list(rows[0].keys())
        write_header = not manifest_path.exists()
        with open(manifest_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=header)
            if write_header:
                writer.writeheader()
            for r in [x for x in rows if x["target"] == pdb_id]:
                writer.writerow(r)

    print(f"\nbuilt {accepted} targets -> {manifest_path}")


if __name__ == "__main__":
    main()

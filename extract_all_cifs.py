# deps: pip install biopython pandas requests
import os, sys, glob, requests, pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

WATER = {"HOH", "WAT"}

def as_list(m, key):
    v = m.get(key)
    if v is None:
        return []
    return v if isinstance(v, list) else [v]

def parse_polymer_chains(m):
    """Return all polymer chain IDs."""
    chains = []
    for s in as_list(m, "_entity_poly.pdbx_strand_id"):
        for ch in (s or "").replace(",", " ").split():
            if ch and ch not in chains:
                chains.append(ch)
    if not chains:
        asym_id = as_list(m, "_struct_asym.id")
        asym_ent = as_list(m, "_struct_asym.entity_id")
        ent_type = as_list(m, "_entity.type")
        ent_map = {str(i+1): ent_type[i] for i in range(len(ent_type))}
        for a, e in zip(asym_id, asym_ent):
            if ent_map.get(str(e), "").lower().startswith("polymer"):
                if a and a not in chains:
                    chains.append(a)
    return chains

def get_smiles(code):
    try:
        r = requests.get(f"https://data.rcsb.org/rest/v1/core/chemcomp/{code.upper()}", timeout=10)
        if r.status_code != 200:
            return ""
        d = r.json().get("rcsb_chem_comp_descriptor", {}) or {}
        return d.get("smiles") or d.get("smiles_stereo") or ""
    except Exception:
        return ""

def process_cif(cif_path, skip_water=True):
    m = MMCIF2Dict(cif_path)
    pdb_id = (as_list(m, "_entry.id")[0] if as_list(m, "_entry.id")
              else os.path.splitext(os.path.basename(cif_path))[0]).upper()
    pdb_path = os.path.abspath(cif_path)
    chains = parse_polymer_chains(m)
    asym   = as_list(m, "_pdbx_nonpoly_scheme.asym_id")
    comp   = as_list(m, "_pdbx_nonpoly_scheme.mon_id")
    authsn = as_list(m, "_pdbx_nonpoly_scheme.auth_seq_num")

    rows = []
    for code, resnum, lig_asym in zip(comp, authsn, asym):
        code = (code or "").upper()
        if not code or (skip_water and code in WATER):
            continue
        lig_smiles = get_smiles(code)
        lig_resi = str(resnum or "")
        for ch in chains:
            rows.append({
                "pdb_id": pdb_id,
                "pdb_path": pdb_path,
                "chain1": ch,
                "chain2": "",
                "lig_code": code,
                "lig_smiles": lig_smiles,
                "lig_resi": lig_resi
            })
    return rows

def main(folder, out_csv):
    all_rows = []
    cif_files = sorted(glob.glob(os.path.join(folder, "*.cif")))
    print(f"Found {len(cif_files)} CIF files in {folder}")
    for i, cif in enumerate(cif_files, 1):
        print(f"[{i}/{len(cif_files)}] {os.path.basename(cif)}")
        try:
            rows = process_cif(cif)
            all_rows.extend(rows)
        except Exception as e:
            print(f"  ⚠️ error processing {cif}: {e}")

    if not all_rows:
        print("No data extracted.")
        return
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"Wrote {len(all_rows)} total rows → {out_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_all_cifs.py <folder> <output.csv>")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])

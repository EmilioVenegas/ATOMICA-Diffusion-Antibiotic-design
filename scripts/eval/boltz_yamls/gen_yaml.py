"""Generate Boltz YAML configs from structured JSON input.

This standalone script replaces the original notebook-oriented version so that
it can be invoked from the command line. Provide a JSON file containing the
sequence, constraint, and template metadata and the script will emit the YAML
format expected by the Boltz CLI.

Example usage:

    poetry run python gen_yaml.py \
        --input config.json \
        --output boltz_job.yaml

The input JSON should roughly mirror the structure used in the original
notebook cell, e.g.

    {
      "job_title": "sample_job",
      "seq_data": [
        {"type": "protein", "chain": "A", "sequence": "...",
         "cyclic": false, "msa": ""}
      ],
      "mod_data": [],
      "bond_data": [],
      "contact_data": [],
      "pocket_data": [],
      "template_data": [],
      "binder": null
    }

Any fields can be omitted if not needed; they will default to empty lists or
``None``. See the README for more context on valid values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

import yaml


POLYMER_TYPES = {"protein", "dna", "rna"}


def _as_chain_lookup(sequences: Iterable[Mapping[str, Any]]) -> Dict[str, str]:
    """Map chain identifiers to their declared type."""

    lookup: Dict[str, str] = {}
    for entry in sequences:
        chain = entry.get("chain")
        seq_type = entry.get("type")
        if chain and seq_type:
            lookup[str(chain)] = str(seq_type)
    return lookup


def _is_polymer(chain_id: str, chain_types: Mapping[str, str]) -> bool:
    return chain_types.get(chain_id, "").lower() in POLYMER_TYPES


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _to_sequences(seq_data: Iterable[Mapping[str, Any]],
                  mod_data: Iterable[Mapping[str, Any]]) -> List[MutableMapping[str, Any]]:
    sequences: List[MutableMapping[str, Any]] = []
    mod_by_chain: Dict[str, List[MutableMapping[str, Any]]] = {}

    for mod in mod_data or []:
        chain = str(mod.get("chain"))
        if chain not in mod_by_chain:
            mod_by_chain[chain] = []
        try:
            index = int(mod["index"])  # type: ignore[index]
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Invalid modification entry: {mod!r}")
        mod_by_chain[chain].append({"position": index, "ccd": mod.get("ccd")})

    for entry in seq_data or []:
        seq_type = str(entry.get("type", "")).lower()
        chain_id = str(entry.get("chain"))
        if not seq_type or not chain_id:
            raise ValueError(f"Sequence entry missing type or chain: {entry!r}")

        if seq_type in POLYMER_TYPES:
            payload: MutableMapping[str, Any] = {
                seq_type: {
                    "id": chain_id,
                    "sequence": entry.get("sequence", ""),
                    "cyclic": bool(entry.get("cyclic", False)),
                }
            }
            if mod_by_chain.get(chain_id):
                payload[seq_type]["modifications"] = mod_by_chain[chain_id]
            msa = entry.get("msa")
            if seq_type == "protein" and msa:
                payload[seq_type]["msa"] = msa
            sequences.append(payload)
        elif seq_type == "smiles":
            sequences.append({"ligand": {"id": chain_id, "smiles": entry.get("sequence", "")}})
        elif seq_type == "ccd":
            sequences.append({"ligand": {"id": chain_id, "ccd": entry.get("sequence", "")}})
        else:
            raise ValueError(f"Unsupported sequence type: {seq_type!r}")

    return sequences


def _add_constraints(output: MutableMapping[str, Any], *, chain_types: Mapping[str, str],
                     bond_data: Iterable[Mapping[str, Any]] | None,
                     pocket_data: Iterable[Mapping[str, Any]] | None,
                     contact_data: Iterable[Mapping[str, Any]] | None,
                     binder: Optional[str]) -> None:
    bonds = list(bond_data or [])
    pockets = list(pocket_data or [])
    contacts = list(contact_data or [])

    if not (bonds or pockets or contacts):
        return

    constraints: List[MutableMapping[str, Any]] = []

    for bond in bonds:
        try:
            atom1 = [bond["chain1"], int(bond["index1"]), bond["atom1"]]
            atom2 = [bond["chain2"], int(bond["index2"]), bond["atom2"]]
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Invalid bond constraint: {bond!r}")
        constraints.append({"bond": {"atom1": atom1, "atom2": atom2}})

    if pockets:
        if not binder:
            raise ValueError("Pocket constraints require a `binder` entry in the input data.")
        pocket_entry: MutableMapping[str, Any] = {
            "pocket": {"binder": binder, "contacts": [], "max_distance": 0.0}
        }
        max_distance = 0.0

        for pocket in pockets:
            chain = str(pocket.get("chain"))
            token_value = pocket.get("token")
            if chain == "None" or chain == "":
                raise ValueError(f"Pocket constraint missing chain identifier: {pocket!r}")
            if _is_polymer(chain, chain_types):
                try:
                    token = int(token_value)
                except (TypeError, ValueError):
                    raise ValueError(f"Expected integer token for polymer chain {chain}: {pocket!r}")
            else:
                token = token_value
            pocket_entry["pocket"]["contacts"].append([chain, token])

            try:
                distance = float(pocket.get("max_d", 0.0))
            except (TypeError, ValueError):
                raise ValueError(f"Invalid max distance in pocket constraint: {pocket!r}")
            max_distance = max(max_distance, distance)

        pocket_entry["pocket"]["max_distance"] = max_distance
        constraints.append(pocket_entry)

    for contact in contacts:
        for key in ("chain1", "chain2"):
            if key not in contact:
                raise ValueError(f"Contact constraint missing {key}: {contact!r}")

        chain1 = str(contact["chain1"])
        chain2 = str(contact["chain2"])
        token1_val = contact.get("token1")
        token2_val = contact.get("token2")

        if _is_polymer(chain1, chain_types):
            try:
                token1 = int(token1_val)
            except (TypeError, ValueError):
                raise ValueError(f"Invalid token1 for contact constraint: {contact!r}")
        else:
            token1 = token1_val

        if _is_polymer(chain2, chain_types):
            try:
                token2 = int(token2_val)
            except (TypeError, ValueError):
                raise ValueError(f"Invalid token2 for contact constraint: {contact!r}")
        else:
            token2 = token2_val

        try:
            max_distance = float(contact.get("max_d", 0.0))
        except (TypeError, ValueError):
            raise ValueError(f"Invalid max distance in contact constraint: {contact!r}")

        constraints.append(
            {
                "contact": {
                    "token1": [chain1, token1],
                    "token2": [chain2, token2],
                    "max_distance": max_distance,
                }
            }
        )

    output["constraints"] = constraints


def _add_templates(output: MutableMapping[str, Any], template_data: Iterable[Mapping[str, Any]] | None) -> None:
    templates = []
    for entry in template_data or []:
        cif_path = entry.get("cif")
        if not cif_path:
            raise ValueError(f"Template entry missing CIF filename: {entry!r}")
        template: MutableMapping[str, Any] = {"cif": str(cif_path)}
        if entry.get("protein"):
            template["chain_id"] = entry["protein"]
        template_id = entry.get("template")
        if template_id and template_id != "Auto":
            template["template_id"] = template_id
        templates.append(template)

    if templates:
        output["templates"] = templates


def build_yaml_structure(raw_input: Mapping[str, Any]) -> Dict[str, Any]:
    seq_data = raw_input.get("seq_data", [])
    mod_data = raw_input.get("mod_data", [])
    bond_data = raw_input.get("bond_data", [])
    pocket_data = raw_input.get("pocket_data", [])
    contact_data = raw_input.get("contact_data", [])
    template_data = raw_input.get("template_data", [])
    binder = raw_input.get("binder")

    sequences = _to_sequences(seq_data, mod_data)
    chain_lookup = _as_chain_lookup(seq_data)

    output: Dict[str, Any] = {"version": 1, "sequences": sequences}

    _add_constraints(
        output,
        chain_types=chain_lookup,
        bond_data=bond_data,
        pocket_data=pocket_data,
        contact_data=contact_data,
        binder=binder,
    )

    _add_templates(output, template_data)

    # Optional affinity property binding selection (mimics previous lig_select widget)
    lig_select = raw_input.get("lig_select")
    if isinstance(lig_select, Mapping):
        lig_value = lig_select.get("value")
    else:
        lig_value = lig_select

    if lig_value and lig_value != "None":
        output["properties"] = [{"affinity": {"binder": lig_value}}]

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Boltz YAML config from JSON input.")
    parser.add_argument("--input", type=Path, required=True, help="Path to JSON file describing the job inputs.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination YAML path. Defaults to <job_title>.yaml or <input-stem>.yaml if no job title is provided.",
    )
    parser.add_argument(
        "--job-title",
        help="Optional job title override used to derive the output filename when --output is not provided.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_input = _load_json(args.input)

    job_title = args.job_title or raw_input.get("job_title")
    output_path = args.output
    if output_path is None:
        if job_title:
            output_path = Path(f"{job_title}.yaml")
        else:
            output_path = args.input.with_suffix(".yaml")

    structure = build_yaml_structure(raw_input)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("w", encoding="utf-8") as handle:
        yaml.dump(structure, handle, default_flow_style=False, sort_keys=False)

    print(f"Wrote YAML config to {output_path}")


if __name__ == "__main__":
    main()
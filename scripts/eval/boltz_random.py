"""Sample random ligands, score them with Boltz-2, and summarise affinity scores.

This script is intended for command line use. Provide a chemical space file (e.g.
CSV with a `smiles` column), optionally a YAML template describing the receptor
setup, and it will:

1. Randomly sample ligands from the chemical space.
2. Materialise Boltz YAML files for each sample (updating the ligand entry).
3. Invoke the Boltz-2 predictor in CPU mode to obtain affinity estimates.
4. Collate the affinity distribution and write summary statistics / histogram.

Example
-------

```bash
poetry run python scripts/eval/random.py \
    --chemical-space data/chemical_space.csv \
    --column smiles \
    --sample-size 50 \
    --template scripts/eval/templates/base_target.yaml \
    --binder-id LIG \
    --output-dir outputs/random_boltz_eval
```

Notes
-----
- Boltz requires substantial compute even on CPU; expect the prediction stage to
  take several minutes depending on `sample_size` and the selected sampling
  parameters.
- The template YAML should contain all non-ligand entries (e.g. protein chains,
  constraints). The ligand entry with id `binder-id` will be replaced per
  sampled molecule. If no template is provided, only a ligand entry is emitted.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping

import numpy as np
import yaml

from boltz.main import predict as boltz_predict_command


def _call_boltz_predict(**kwargs) -> None:
    """Invoke the Boltz CLI predict command programmatically."""

    callback = getattr(boltz_predict_command, "callback", None)
    if callback is None:
        # Fallback in case the import already returned the raw function
        callback = boltz_predict_command

    callback(**kwargs)


DEFAULT_BINDER_ID = "LIG"


def _read_smiles_from_csv(path: Path, column: str) -> List[str]:
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover - pandas is in boltz deps
        msg = "pandas is required to read CSV chemical space files"
        raise RuntimeError(msg) from exc

    df = pd.read_csv(path)
    if column not in df.columns:
        available = ", ".join(df.columns)
        raise ValueError(
            f"Column '{column}' not found in {path}. Available columns: {available}"
        )
    smiles = df[column].dropna().astype(str).tolist()
    if not smiles:
        raise ValueError(f"Column '{column}' in {path} does not contain any SMILES entries")
    return smiles


def _read_smiles_from_plain_text(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        smiles = [line.strip() for line in handle if line.strip()]
    if not smiles:
        raise ValueError(f"No molecules found in {path}")
    return smiles


def load_chemical_space(path: Path, column: str | None = None) -> List[str]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt"}:
        sep = "," if suffix == ".csv" else "\t"
        if suffix in {".csv", ".tsv"}:
            col = column or "smiles"
            return _read_smiles_from_csv(path, col)
        # Fallback to plain text for .txt
        return _read_smiles_from_plain_text(path)
    elif suffix in {".smi", ""}:  # allow extension-less plain text
        return _read_smiles_from_plain_text(path)
    else:
        raise ValueError(f"Unsupported chemical-space format: {path.suffix}")


def load_template(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {"version": 1, "sequences": []}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {"version": 1, "sequences": []}


def _normalise_sequences(data: MutableMapping[str, Any]) -> None:
    sequences = data.setdefault("sequences", [])
    if not isinstance(sequences, list):
        raise TypeError("'sequences' must be a list in the template data")


def _replace_ligand_entry(
    data: MutableMapping[str, Any],
    *,
    binder_id: str,
    smiles: str,
) -> None:
    _normalise_sequences(data)
    sequences = data["sequences"]

    # Remove existing ligand entries with the same id
    filtered_sequences = []
    for entry in sequences:
        if not isinstance(entry, Mapping):
            continue
        ligand_block = entry.get("ligand")
        if ligand_block and isinstance(ligand_block, Mapping):
            if str(ligand_block.get("id")) == binder_id:
                continue  # skip existing ligand entry, we will replace it
        filtered_sequences.append(entry)

    filtered_sequences.append({"ligand": {"id": binder_id, "smiles": smiles}})
    data["sequences"] = filtered_sequences


def _ensure_affinity_property(data: MutableMapping[str, Any], binder_id: str) -> None:
    properties = data.setdefault("properties", [])
    if not isinstance(properties, list):
        raise TypeError("'properties' section must be a list")

    affinity_entry = {"affinity": {"binder": binder_id}}

    for idx, entry in enumerate(properties):
        if not isinstance(entry, Mapping):
            continue
        affinity_block = entry.get("affinity")
        if affinity_block and isinstance(affinity_block, Mapping):
            properties[idx] = affinity_entry
            break
    else:
        properties.append(affinity_entry)


def materialise_yaml_samples(
    smiles_list: Iterable[str],
    *,
    template: Mapping[str, Any],
    binder_id: str,
    output_dir: Path,
) -> List[Path]:
    yaml_paths: List[Path] = []
    for idx, smiles in enumerate(smiles_list, start=1):
        sample_id = f"sample_{idx:04d}"
        yaml_data = deepcopy(template)
        yaml_data.setdefault("version", 1)

        _replace_ligand_entry(yaml_data, binder_id=binder_id, smiles=smiles)
        _ensure_affinity_property(yaml_data, binder_id=binder_id)

        yaml_path = output_dir / f"{sample_id}.yaml"
        with yaml_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(yaml_data, handle, sort_keys=False)
        yaml_paths.append(yaml_path)
    return yaml_paths


def run_boltz_predictions(
    *,
    yaml_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    accelerator: str,
    sampling_steps: int,
    diffusion_samples: int,
    sampling_steps_affinity: int,
    diffusion_samples_affinity: int,
) -> Path:
    _call_boltz_predict(
        data=str(yaml_dir),
        out_dir=str(output_dir),
        cache=str(cache_dir),
        accelerator=accelerator,
        devices=1,
        recycling_steps=1,
        sampling_steps=sampling_steps,
        diffusion_samples=diffusion_samples,
        sampling_steps_affinity=sampling_steps_affinity,
        diffusion_samples_affinity=diffusion_samples_affinity,
        max_parallel_samples=1,
        num_workers=0,
        override=True,
        use_msa_server=False,
        model="boltz2",
        method="other",
        subsample_msa=False,
        no_kernels=True,
    )

    result_root = output_dir / f"boltz_results_{yaml_dir.stem}"
    return result_root


def collect_affinity_scores(prediction_root: Path) -> Dict[str, Dict[str, float]]:
    predictions_dir = prediction_root / "predictions"
    if not predictions_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")

    affinity_scores: Dict[str, Dict[str, float]] = {}
    for sample_dir in predictions_dir.iterdir():
        if not sample_dir.is_dir():
            continue
        affinity_json = sample_dir / f"affinity_{sample_dir.name}.json"
        if not affinity_json.exists():
            continue
        with affinity_json.open("r", encoding="utf-8") as handle:
            affinity_scores[sample_dir.name] = json.load(handle)

    if not affinity_scores:
        raise RuntimeError(
            f"No affinity predictions were found under {predictions_dir}. "
            "Check that the YAML inputs requested affinity properties."
        )

    return affinity_scores


def summarise_scores(scores: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    values = np.array([entry["affinity_pred_value"] for entry in scores.values()], dtype=float)
    probabilities = np.array(
        [entry["affinity_probability_binary"] for entry in scores.values()], dtype=float
    )

    summary = {
        "count": len(values),
        "affinity_pred_value": {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
            "std": float(values.std(ddof=0)),
        },
        "affinity_probability_binary": {
            "mean": float(probabilities.mean()),
            "median": float(np.median(probabilities)),
            "min": float(probabilities.min()),
            "max": float(probabilities.max()),
            "std": float(probabilities.std(ddof=0)),
        },
    }

    hist_counts, bin_edges = np.histogram(values, bins="auto")
    summary["histogram"] = {
        "bin_edges": bin_edges.tolist(),
        "counts": hist_counts.tolist(),
    }

    return summary


def write_summary(output_dir: Path, *, scores: Dict[str, Dict[str, float]], summary: Dict[str, Any]) -> None:
    (output_dir / "summaries").mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "summaries" / "affinity_scores.json"
    summary_path = output_dir / "summaries" / "affinity_summary.json"

    with score_path.open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=4)

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chemical-space", type=Path, required=True, help="Path to the chemical space file (CSV/SMI/TXT)")
    parser.add_argument("--column", default="smiles", help="Column name containing SMILES when loading from CSV/TSV (default: smiles)")
    parser.add_argument("--sample-size", type=int, default=32, help="Number of molecules to sample")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--template", type=Path, help="Template YAML describing the target (optional)")
    parser.add_argument("--binder-id", default=DEFAULT_BINDER_ID, help="Ligand/binder identifier to overwrite in the template (default: LIG)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/boltz_random"), help="Directory where YAMLs, predictions, and summaries are stored")
    parser.add_argument("--cache-dir", type=Path, default=Path("~/.boltz").expanduser(), help="Boltz model cache directory")
    parser.add_argument("--accelerator", choices={"cpu", "gpu"}, default="cpu", help="Hardware accelerator for Boltz predictions (default: cpu)")
    parser.add_argument("--sampling-steps", type=int, default=25, help="Diffusion sampling steps for structure prediction (smaller is faster)")
    parser.add_argument("--diffusion-samples", type=int, default=1, help="Number of diffusion samples for structure prediction")
    parser.add_argument("--sampling-steps-affinity", type=int, default=50, help="Sampling steps used by the affinity head")
    parser.add_argument("--diffusion-samples-affinity", type=int, default=1, help="Diffusion samples for affinity prediction")
    parser.add_argument("--keep-inputs", action="store_true", help="Retain the temporary YAML input directory after prediction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_random_affinity_workflow(
        chemical_space=args.chemical_space,
        sample_size=args.sample_size,
        column=args.column,
        seed=args.seed,
        template_path=args.template,
        output_dir=args.output_dir,
        binder_id=args.binder_id,
        cache_dir=args.cache_dir,
        accelerator=args.accelerator,
        sampling_steps=args.sampling_steps,
        diffusion_samples=args.diffusion_samples,
        sampling_steps_affinity=args.sampling_steps_affinity,
        diffusion_samples_affinity=args.diffusion_samples_affinity,
        keep_inputs=args.keep_inputs,
    )


def run_random_affinity_workflow(
    *,
    chemical_space: Path,
    sample_size: int,
    column: str | None,
    seed: int | None,
    template_path: Path | None,
    output_dir: Path,
    binder_id: str,
    cache_dir: Path,
    accelerator: str,
    sampling_steps: int,
    diffusion_samples: int,
    sampling_steps_affinity: int,
    diffusion_samples_affinity: int,
    keep_inputs: bool,
) -> Dict[str, Any]:
    smiles_space = load_chemical_space(chemical_space, column)
    if sample_size > len(smiles_space):
        raise ValueError(
            f"Sample size {sample_size} exceeds the number of available molecules ({len(smiles_space)})"
        )

    rng = random.Random(seed)
    sampled_smiles = rng.sample(smiles_space, sample_size)

    template = load_template(template_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_dir = output_dir / "inputs"
    if yaml_dir.exists():
        shutil.rmtree(yaml_dir)
    yaml_dir.mkdir(parents=True)

    materialise_yaml_samples(
        sampled_smiles,
        template=template,
        binder_id=binder_id,
        output_dir=yaml_dir,
    )

    prediction_root = run_boltz_predictions(
        yaml_dir=yaml_dir,
        output_dir=output_dir,
        cache_dir=cache_dir,
        accelerator=accelerator,
        sampling_steps=sampling_steps,
        diffusion_samples=diffusion_samples,
        sampling_steps_affinity=sampling_steps_affinity,
        diffusion_samples_affinity=diffusion_samples_affinity,
    )

    scores = collect_affinity_scores(prediction_root)
    summary = summarise_scores(scores)
    write_summary(output_dir, scores=scores, summary=summary)

    if not keep_inputs:
        shutil.rmtree(yaml_dir, ignore_errors=True)

    print(json.dumps(summary, indent=2))  # noqa: T201
    return summary


if __name__ == "__main__":
    main()


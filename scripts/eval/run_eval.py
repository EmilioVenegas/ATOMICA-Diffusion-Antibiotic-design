from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import yaml


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError("Could not locate project root containing 'scripts/'")


initial_dir = Path.cwd().resolve()
PROJECT_ROOT = find_project_root(initial_dir)
if initial_dir != PROJECT_ROOT:
    os.chdir(PROJECT_ROOT)

BASE_DIR = Path.cwd()
DUMMY_DIR = BASE_DIR / "scripts" / "eval" / "dummy_data"
DUMMY_DIR.mkdir(parents=True, exist_ok=True)
CHEM_SPACE_PATH = DUMMY_DIR / "dummy_chemical_space.csv"
TEMPLATE_PATH = DUMMY_DIR / "dummy_template.yaml"
OUTPUT_DIR = DUMMY_DIR / "demo_outputs"

print(f"Initial working directory: {initial_dir}")
print(f"Project root (cwd set to): {BASE_DIR}")
print(f"Dummy data directory: {DUMMY_DIR}")

dummy_smiles = pd.DataFrame(
    {
        "smiles": [
            "CCO",  # ethanol
            "CC(=O)O",  # acetic acid
            "c1ccccc1",  # benzene
            "C1CCCCC1",  # cyclohexane
            "CCN(CC)CC",  # triethylamine
            "CCOC(=O)C",  # ethyl acetate
            "CC(C)O",  # isopropanol
            "CC(C)=O",  # acetone
            "C=CCO",  # allyl alcohol
            "CC(C)C",  # isobutane
        ]
    }
)
dummy_smiles.to_csv(CHEM_SPACE_PATH, index=False)

print(f"Wrote {CHEM_SPACE_PATH} with {len(dummy_smiles)} molecules")


template_yaml = {
    "version": 1,
    "sequences": [
        {
            "protein": {
                "id": "A",
                "sequence": "MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQKDWLDNEFGYSNWSKIDDEI"
                "DDN",  # short toy sequence, replace with real target
                "msa": "empty"
            }
        }
    ],
    "properties": [
        {"affinity": {"binder": "LIG"}}
    ],
}

with TEMPLATE_PATH.open("w", encoding="utf-8") as handle:
    import yaml

    yaml.safe_dump(template_yaml, handle, sort_keys=False)

print(f"Wrote template to {TEMPLATE_PATH}")


import importlib
import scripts.eval.random as random_module

importlib.reload(random_module)

summary = random_module.run_random_affinity_workflow(
    chemical_space=CHEM_SPACE_PATH,
    sample_size=10,
    column="smiles",
    seed=42,
    template_path=TEMPLATE_PATH,
    output_dir=OUTPUT_DIR,
    binder_id="LIG",
    cache_dir=Path("~/.boltz").expanduser(),
    accelerator="cpu",
    sampling_steps=10,
    diffusion_samples=1,
    sampling_steps_affinity=20,
    diffusion_samples_affinity=1,
    keep_inputs=True,
)

print(json.dumps(summary, indent=2))

config_yaml = {
    "version": 1,
    "sequences": [
        {
            "protein": {
                "id": "A",
                "sequence": "MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQKDWLDNEFGYSNWSKIDDEI"
                "DDN",
                "msa": "empty",
            }
        }
    ],
    "properties": [
        {"affinity": {"binder": "LIG"}}
    ],
}

with TEMPLATE_PATH.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(config_yaml, handle, sort_keys=False)

print(f"Wrote Boltz config to {TEMPLATE_PATH}")

summary_path = OUTPUT_DIR / "summaries" / "affinity_summary.json"

if summary_path.exists():
    with summary_path.open("r", encoding="utf-8") as handle:
        summary_loaded = json.load(handle)
    print("Loaded summary from disk:")
    print(json.dumps(summary_loaded, indent=2))
else:
    print(f"Summary file not found at {summary_path}. Have you run the prediction cell?")


import numpy as np
import matplotlib.pyplot as plt

summary_path = OUTPUT_DIR / "summaries" / "affinity_summary.json"
if summary_path.exists():
    with summary_path.open("r", encoding="utf-8") as handle:
        affinity_summary = json.load(handle)
    scores_path = OUTPUT_DIR / "summaries" / "affinity_scores.json"
    if scores_path.exists():
        with scores_path.open("r", encoding="utf-8") as handle:
            affinity_scores = json.load(handle)
        values = np.array([entry["affinity_pred_value"] for entry in affinity_scores.values()])
        plt.figure(figsize=(6, 4))
        plt.hist(values, bins=min(10, len(values)), color="steelblue", edgecolor="white")
        plt.xlabel("affinity_pred_value")
        plt.ylabel("Count")
        plt.title("Boltz-2 Affinity Distribution")
        plt.show()
    else:
        print("Affinity scores file not found; run the prediction cell first.")
else:
    print("Affinity summary not found; run the prediction cell first.")

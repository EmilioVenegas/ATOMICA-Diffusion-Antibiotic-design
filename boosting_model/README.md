## 🧬 `boosting_model/` — Developer Overview

### 1. Purpose

This module implements the **Ligand–Protein Boosting baseline**, used for the *Dynamic properties scoring* portion of the project to predict -log(pKa) prior to generative diffusion within the ATOMICA–Diffusion pipeline. 

---

### 2. File and Folder Descriptions

#### 📄 `analyze_baseline.py`

* Performs post-training analysis of the boosting model outputs.
* Loads stored predictions and metrics from `artifacts_baseline/`.
* Computes standard regression metrics (RMSE, R², Spearman correlation).
* Generates visualizations such as feature importance and performance plots.
* Example usage:

  ```bash
  python analyze_baseline.py --results artifacts_baseline/results.csv
  ```

#### 📄 `chembl_boosting_baseline.py`

* Core training script for the ligand–protein affinity boosting model.
* Loads curated ChEMBL bioactivity data (`Kd`, `Ki`, or `IC₅₀`).
* Uses **Morgan Fingerprints (radius=2, 1024 bits)** plus 8 physicochemical descriptors.
* Trains an **XGBoost/LightGBM** regressor with 5-fold cross-validation.
* Saves model weights, logs, and metrics in `artifacts_baseline/`.

#### 📘 `Ligand_Protein_Boosting.ipynb`

* This file contains interpretation of results thus far and questions to be answered before any further modeling

#### 📁 `artifacts_baseline/`

* Stores experiment artifacts, including:

  * Metrics and logs
  * Generated visualizations 

#### 📄 `requirements.txt`

* Python dependencies required for this submodule:

  ```
  pandas
  scikit-learn
  numpy
  matplotlib
  rdkit
  ...
  ```
* Install using:

  ```bash
  pip install -r boosting_model/requirements.txt
  ```

---

### 3. Usage Example

```bash
# Activate environment
micromamba activate chembl_env

# Train baseline model
python boosting_model/chembl_boosting_baseline.py

# Analyze performance
python boosting_model/analyze_baseline.py
```

---

### 4. Development Notes

* Keep artifacts and large datasets **out of version control**.
* Update this README when introducing new experiments or files.
* Use versioned commits (e.g., `boost:`, `fix:`, `doc:`) for clarity.
* Tag milestone versions, e.g., `v0.1.0-baseline`.

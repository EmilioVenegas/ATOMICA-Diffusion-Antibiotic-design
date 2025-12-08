# RL Loop Quick Start Guide

## 🚀 5-Minute Setup

### 1. Install Dependencies

```bash
# Using conda (recommended if you have DiffSBDD environment)
conda activate diffsbdd
pip install admet-ai pyarrow

# OR using pip only
pip install rdkit-pypi admet-ai pandas pyarrow
```

### 2. Test with Example Ligands

```bash
# From the repo root directory
cd rl_loop

# Run on DiffSBDD examples
python RL_loop.py -i ../DiffSBDD/example -o results/example_scores.csv

# View results
python -c "import pandas as pd; df=pd.read_csv('results/example_scores.csv'); print(df.head())"
```

### 3. Run the Test Suite

```bash
python test_rl_loop.py
```

## 📊 Understanding the Output

The output CSV contains:

```csv
rank,molecule_id,smiles,source_file,composite_score,[ADMET properties...]
1,3rfm_B_CFF.sdf_0,Cc1nc2ccccc2...,3rfm_B_CFF.sdf,0.742,...
2,5ndu_C_8V2.sdf_0,COc1ccc2nc...,5ndu_C_8V2.sdf,0.691,...
```

### Key Columns:
- **rank**: Overall ranking (1 = best)
- **molecule_id**: Unique identifier
- **smiles**: SMILES representation
- **composite_score**: Weighted average of ADMET properties (0-1)
- **ADMET properties**: Individual predictions (solubility, clearance, toxicity, etc.)

## 🔄 Typical Workflow

### Step 1: Generate Ligands
```bash
cd ../DiffSBDD
python generate_ligands.py checkpoints/model.ckpt \
  --pdbfile example/3rfm.pdb \
  --outfile ../rl_loop/examples/generated.sdf \
  --ref_ligand A:330 \
  --n_samples 100
```

### Step 2: Score and Rank
```bash
cd ../rl_loop
python RL_loop.py \
  -i examples/generated.sdf \
  -o results/scored_generated.csv
```

### Step 3: Select Top Candidates
```bash
# Keep only top 10
python RL_loop.py \
  -i examples/generated.sdf \
  -o results/top10.csv \
  -k 10
```

### Step 4: Analyze Results
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('results/scored_generated.csv')

# View top molecules
print(df.head(10))

# Plot score distribution
df['composite_score'].hist(bins=20)
plt.xlabel('Composite Score')
plt.ylabel('Count')
plt.title('ADMET Score Distribution')
plt.show()

# Export top SMILES for further analysis
top_smiles = df.head(20)['smiles'].tolist()
with open('results/top20_smiles.txt', 'w') as f:
    f.write('\n'.join(top_smiles))
```

## 🔧 Common Issues

### ADMET-AI Not Found
```bash
pip install admet-ai
```

### RDKit Import Error
```bash
# If using pip
pip install rdkit-pypi

# If using conda
conda install -c conda-forge rdkit
```

### Invalid SDF File
- Check that your SDF file is valid: `obabel input.sdf -osdf -O validated.sdf`
- Some molecules may fail sanitization - this is normal

## 📈 Next Steps

1. **Add SA Score**: Integrate synthetic accessibility scoring
2. **Add Brenk Alerts**: Flag molecules with structural alerts
3. **Custom Scoring**: Define your own composite score function
4. **Iterative Generation**: Use scores to guide next round of generation

## 🐛 Debugging

Run in verbose mode (default) to see detailed progress:
```bash
python RL_loop.py -i examples/ -o results/debug.csv
```

Run in quiet mode:
```bash
python RL_loop.py -i examples/ -o results/output.csv --quiet
```

Check the test script output:
```bash
python test_rl_loop.py > test_output.txt 2>&1
```

## 💡 Pro Tips

1. **Batch Processing**: Process large files in chunks to avoid memory issues
2. **Property Selection**: Focus on specific ADMET properties relevant to your target
3. **Normalization**: Consider normalizing scores based on known reference compounds
4. **Visualization**: Use tools like RDKit or ChemDraw to visualize top candidates


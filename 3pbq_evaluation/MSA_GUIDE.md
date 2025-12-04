# Generating MSA for 3PBQ using HH-suite

## Prerequisites

1. **Install HH-suite**:
   ```bash
   # Option 1: Using conda/micromamba (recommended)
   conda install -c bioconda hhsuite
   
   # Option 2: Using Homebrew (macOS)
   brew install hhsuite
   
   # Option 3: Download from GitHub
   # https://github.com/soedinglab/hh-suite
   ```

2. **Download a protein database**:
   
   **Option A: UniRef30** (Smaller, ~50-70GB, recommended for smaller storage)
   - Download from: https://wwwuser.gwdg.de/~compbiol/uniclust/current_release/
   - Direct link (example): http://wwwuser.gwdg.de/~compbiol/uniclust/2023_02/UniRef30_2023_02_hhsuite.tar.gz
   ```bash
   # UniRef30 (~50-70GB compressed, ~100-150GB uncompressed)
   wget http://wwwuser.gwdg.de/~compbiol/uniclust/2023_02/UniRef30_2023_02_hhsuite.tar.gz
   tar -xzf UniRef30_2023_02_hhsuite.tar.gz
   ```
   
   **Option B: UniClust30** (Larger, ~100GB, more comprehensive)
   - Download from: https://uniclust.mmseqs.com/
   ```bash
   # UniClust30 (~100GB compressed, ~200GB uncompressed)
   wget http://wwwuser.gwdg.de/~compbiol/uniclust/2023_02/UniClust30_2023_02_hhsuite.tar.gz
   tar -xzf UniClust30_2023_02_hhsuite.tar.gz
   ```
   
   **Note**: UniRef30 is smaller and sufficient for most use cases. UniClust30 is more comprehensive but larger.

## Generate MSA

### Method 1: Using the provided script

```bash
# Set database path (use the path where you extracted the database)
export HHSUITE_DB=/path/to/UniRef30_2023_02  # or UniClust30_2023_02

# Make script executable
chmod +x generate_msa.sh

# Run script
./generate_msa.sh
```

### Method 2: Manual command

```bash
cd 3pbq_evaluation
mkdir -p inputs/msa

hhblits \
    -i inputs/3PBQ_sequence.fasta \
    -d /path/to/UniRef30_2023_02 \
    -o inputs/msa/3PBQ.a3m \
    -cpu 4 \
    -maxfilt 50000 \
    -neffmax 20 \
    -cov 25 \
    -id 99
```

## Update YAML Template

After generating the MSA, update `inputs/3pbq_template.yaml`:

```yaml
version: 1
sequences:
- protein:
    id: A
    sequence: VRHIAIPAHRGLITDRNGEPLAVSTPVTTLWANPKELMTAKERWPQLAAALGQDTKLFADRIEQNAEREFIYLVRGLTPEQGEGVIALKVPGVYSIEEFRRFYPAGEVVAHAVGFTDVDDRGREGIELAFDEWLAGVPGKRQVLKDRVQVTKNAKPGKTLALSIDLRLQYLAHRELRNALLENGAKAGSLVIMDVKTGEILAMTNQPTYNPNNRRNLQPAAMRNRAMIDVFEPGSTVKPFSMSAALASGRWKPSDIVDVYPGTLQIGRYTIRDVSRNSRQLDLTGILIKSSNVGISKIAFDIGAESIYSVMQQVGLGQDTGLGFPGERVGNLPNHRKWPKAETATLAYGYGLSVTAIQLAHAYAALANDGKSVPLSMTRVDRVPDGVQVISPEVASTVQGMLQQVVEAQGGVFRAQVPGYHAAGKSGTARKNAYRSLFAGFAPATDPRIAMVVVIDEPSKAGYFGGLVSAPVFSKVMAGALRLMNVPPDNLPT
    msa: "inputs/msa/3PBQ.a3m"  # Update this line
- ligand:
    id: LIG
    smiles: "{{SMILES}}"
properties:
- affinity:
    binder: LIG
```

## Alternative: Use ColabFold (No Local Installation)

If you don't want to install HH-suite locally, you can use ColabFold:

1. Go to: https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb
2. Upload `inputs/3PBQ_sequence.fasta`
3. Run the notebook to generate MSA
4. Download the `.a3m` file
5. Place it in `inputs/msa/3PBQ.a3m`

## Notes

- **Database Size Comparison**:
  - UniRef30: ~50-70GB compressed, ~100-150GB uncompressed (recommended for smaller storage)
  - UniClust30: ~100GB compressed, ~200GB uncompressed (more comprehensive)
- MSA generation can take 10-30 minutes depending on your hardware
- The database only needs to be downloaded once and can be reused for multiple proteins
- For faster results, you can use smaller databases or reduce search parameters
- The generated `.a3m` file will be used by Boltz-2 for more accurate predictions
- Both databases work well with HH-suite; UniRef30 is sufficient for most use cases


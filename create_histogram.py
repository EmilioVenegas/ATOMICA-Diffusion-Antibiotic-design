import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
import sys

data_dir = Path('data/processed_atomica/train')
output_file = Path('data/processed_atomica/size_distribution.npy')

if not data_dir.is_dir():
    print(f"Error: Data directory not found at {data_dir}")
    print("Please make sure you have run Phase 1 and the .pt files are in")
    print("data/processed_atomica/train/")
    sys.exit(1)

# Find all .pt files
files = sorted(data_dir.glob('*.pt'))
if not files:
    print(f"Error: No .pt files found in {data_dir}")
    sys.exit(1)

# Get all ligand sizes
lig_sizes = []
for f in tqdm(files, desc="Processing files to get ligand sizes"):
    try:
        data = torch.load(f)
        lig_sizes.append(len(data['lig_coords']))
    except Exception as e:
        print(f"Warning: Could not load file {f}. Skipping. Error: {e}")

# Create the histogram
max_size = max(lig_sizes)
# We create a 1D histogram of just ligand sizes
histogram = np.zeros(max_size + 1, dtype=int)
for size in lig_sizes:
    histogram[size] += 1

# Save the histogram
np.save(output_file, histogram)
print(f"\nSuccessfully saved histogram to {output_file}")
print(f"Total complexes found: {len(lig_sizes)}")
print(f"Max ligand size: {max_size}")
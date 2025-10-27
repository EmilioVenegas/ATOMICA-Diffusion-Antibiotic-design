from itertools import accumulate
import numpy as np
import torch
from torch.utils.data import Dataset
# NEW: Import for loading .pt files
import os


class ProcessedLigandPocketDataset(Dataset):
    def __init__(self, data_path, center=True, transform=None): # MODIFIED: data_path can be .npz or dir

        self.transform = transform
        
        # MODIFIED: Handle both .npz file (old) and directory of .pt files (new)
        if os.path.isfile(data_path) and data_path.endswith('.npz'):
            print(f"Loading data from NPZ file: {data_path}")
            with np.load(data_path, allow_pickle=True) as f:
                data = {key: val for key, val in f.items()}
            
            # --- Original NPZ loading logic ---
            self.data = {}
            for (k, v) in data.items():
                if k == 'names' or k == 'receptors':
                    self.data[k] = v
                    continue

                sections = np.where(np.diff(data['lig_mask']))[0] + 1 \
                    if 'lig' in k \
                    else np.where(np.diff(data['pocket_mask']))[0] + 1
                self.data[k] = [torch.from_numpy(x) for x in np.split(v, sections)]

                # add number of nodes for convenience
                if k == 'lig_mask':
                    self.data['num_lig_atoms'] = \
                        torch.tensor([len(x) for x in self.data['lig_mask']])
                elif k == 'pocket_mask':
                    self.data['num_pocket_nodes'] = \
                        torch.tensor([len(x) for x in self.data['pocket_mask']])
            
            # Manually add empty pocket_atomica_embeddings if loading old data
            if 'pocket_atomica_embeddings' not in self.data:
                print("Warning: 'pocket_atomica_embeddings' not found. Adding empty list.")
                self.data['pocket_atomica_embeddings'] = [torch.empty(0) for _ in range(len(self.data['names']))]
            
            self.num_samples = len(self.data['names'])
            
        elif os.path.isdir(data_path):
            print(f"Loading data from directory of .pt files: {data_path}")
            self.data_files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.pt')])
            self.num_samples = len(self.data_files)
            print(f"Found {self.num_samples} processed .pt files.")
            self.data_mode = 'pt'
        
        else:
            raise FileNotFoundError(f"Data path not found or is not a .npz file or directory: {data_path}")

        self.center = center # Centering will be applied in __getitem__ if needed

    def __len__(self):
        return self.num_samples # MODIFIED

    def __getitem__(self, idx):
        # MODIFIED: Load data based on mode
        if hasattr(self, 'data_mode') and self.data_mode == 'pt':
            data = torch.load(self.data_files[idx])
            
            # Create masks and sizes on the fly
            data['lig_mask'] = torch.zeros(len(data['lig_coords'])) # Will be replaced by collate_fn
            data['pocket_mask'] = torch.zeros(len(data['pocket_coords'])) # Will be replaced by collate_fn
            data['num_lig_atoms'] = len(data['lig_coords'])
            data['num_pocket_nodes'] = len(data['pocket_coords'])
            
            # Ensure compatibility with existing keys
            # 'pocket_atomica_embeddings' is already loaded
            # 'lig_one_hot' is already loaded

        else: # Original .npz logic
            data = {key: val[idx] for key, val in self.data.items()}

        if self.center: # MODIFIED: Apply centering here
            mean = (data['lig_coords'].sum(0) +
                    data['pocket_coords'].sum(0)) / \
                   (len(data['lig_coords']) + len(data['pocket_coords']))
            data['lig_coords'] = data['lig_coords'] - mean
            data['pocket_coords'] = data['pocket_coords'] - mean

        if self.transform is not None:
            data = self.transform(data)
        return data

    @staticmethod
    def collate_fn(batch):
        out = {}
        for prop in batch[0].keys():

            if prop == 'names' or prop == 'receptors' or prop == 'name':
                out[prop] = [x[prop] for x in batch]
            elif prop == 'num_lig_atoms' or prop == 'num_pocket_nodes' \
                    or prop == 'num_virtual_atoms':
                out[prop] = torch.tensor([x[prop] for x in batch])
            elif 'mask' in prop:
                # make sure indices in batch start at zero (needed for
                # torch_scatter)
                out[prop] = torch.cat([i * torch.ones(len(x[prop.replace('mask', 'coords')])) # MODIFIED: base mask on coords
                                       for i, x in enumerate(batch)], dim=0).long() # MODIFIED: ensure long type
            else:
                # This will now correctly collate 'lig_coords', 'lig_one_hot',
                # 'pocket_coords', and 'pocket_atomica_embeddings'
                out[prop] = torch.cat([x[prop] for x in batch], dim=0)

        return out
from itertools import accumulate
import numpy as np
import torch
from torch.utils.data import Dataset
# NEW: Import for loading .pt files
import os


class ProcessedLigandPocketDataset(Dataset):
    def __init__(self, data_path, center=True, transform=None): # MODIFIED: data_path can be .npz or dir

        self.transform = transform
        
        # MODIFIED: Only handle directory of .pt files
        if os.path.isdir(data_path):
            print(f"Loading data from directory of .pt files: {data_path}")
            self.data_files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.pt')])
            self.num_samples = len(self.data_files)
            print(f"Found {self.num_samples} processed .pt files.")
            self.data_mode = 'pt'
        else:
            raise FileNotFoundError(f"Data path not found or is not a directory: {data_path}")

        self.center = center # Centering will be applied in __getitem__ if needed

    def __len__(self):
        return self.num_samples # MODIFIED

    def __getitem__(self, idx):
        # MODIFIED: Load data based on mode
        # MODIFIED: Load data from .pt file
        data = torch.load(self.data_files[idx])
        
        # Create masks and sizes on the fly
        data['lig_mask'] = torch.zeros(len(data['lig_coords'])) # Will be replaced by collate_fn
        data['pocket_mask'] = torch.zeros(len(data['pocket_coords'])) # Will be replaced by collate_fn
        data['num_lig_atoms'] = len(data['lig_coords'])
        data['num_pocket_nodes'] = len(data['pocket_coords'])
        
        # Ensure compatibility with existing keys
        # 'pocket_atomica_embeddings' is already loaded
        # 'lig_one_hot' is already loaded

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
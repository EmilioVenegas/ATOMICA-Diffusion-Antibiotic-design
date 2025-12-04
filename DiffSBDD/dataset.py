from itertools import accumulate
import numpy as np
import torch
from torch.utils.data import Dataset


class ProcessedLigandPocketDataset(Dataset):
    def __init__(self, npz_path, center=True, transform=None):

        self.transform = transform

        with np.load(npz_path, allow_pickle=True) as f:
            data = {key: val for key, val in f.items()}

        # split data based on mask
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

        if center:
            for i in range(len(self.data['lig_coords'])):
                mean = (self.data['lig_coords'][i].sum(0) +
                        self.data['pocket_coords'][i].sum(0)) / \
                       (len(self.data['lig_coords'][i]) + len(self.data['pocket_coords'][i]))
                self.data['lig_coords'][i] = self.data['lig_coords'][i] - mean
                self.data['pocket_coords'][i] = self.data['pocket_coords'][i] - mean

    def __len__(self):
        return len(self.data['names'])

    def __getitem__(self, idx):
        data = {key: val[idx] for key, val in self.data.items()}
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
                out[prop] = torch.cat([i * torch.ones(len(x[prop]))
                                       for i, x in enumerate(batch)], dim=0)
            else:
                out[prop] = torch.cat([x[prop] for x in batch], dim=0)

        return out

class LigandPocketDatasetPT(Dataset):
    def __init__(self, data_dir, transform=None, center=True):
        self.transform = transform
        self.center = center
        from pathlib import Path
        self.data_dir = Path(data_dir)
        self.files = sorted(list(self.data_dir.glob('*.pt')))
        self.cache = {}
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        if idx in self.cache:
            data = self.cache[idx]
        else:
            data = torch.load(self.files[idx])
            
            # Add size info
            data['num_lig_atoms'] = torch.tensor(data['lig_coords'].shape[0])
            data['num_pocket_nodes'] = torch.tensor(data['pocket_coords'].shape[0])
            
            # Add masks (all zeros for single item, collate_fn will adjust)
            data['lig_mask'] = torch.zeros(data['num_lig_atoms'], dtype=torch.long)
            data['pocket_mask'] = torch.zeros(data['num_pocket_nodes'], dtype=torch.long)
            
            # Ensure types
            data['lig_coords'] = data['lig_coords'].float()
            data['pocket_coords'] = data['pocket_coords'].float()
            data['lig_one_hot'] = data['lig_one_hot'].float()
            data['pocket_one_hot'] = data['pocket_one_hot'].float()
            
            if 'pocket_atomica_embeddings' in data:
                data['pocket_atomica_embeddings'] = data['pocket_atomica_embeddings'].float()

            # Check CoM before centering
            joint_coords = torch.cat([data['lig_coords'], data['pocket_coords']], dim=0)
            com = joint_coords.mean(dim=0)
            
            if self.center:
                # Force centering
                data['lig_coords'] = data['lig_coords'] - com
                data['pocket_coords'] = data['pocket_coords'] - com
            elif com.norm() > 5.0:
                print(f"Warning: Data {self.files[idx]} is far from origin (CoM: {com.numpy()}) but centering is disabled!")
                
            self.cache[idx] = data

        if self.transform is not None:
            data = self.transform(data)
            
        return data
        
    @staticmethod
    def collate_fn(batch):
        return ProcessedLigandPocketDataset.collate_fn(batch)

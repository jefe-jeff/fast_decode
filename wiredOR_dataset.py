import h5py
import helpers
import numpy as np
import torch
from torch.utils import data

class WiredORDataset(data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, file_path, window, chunk_size = 100000, transform=None):
        super().__init__()
        self.fp = file_path
        self.chunk_size = chunk_size
        self.transform = transform
        self.window = window
        
        # Search for all h5 files
        #p = Path(file_path)
        #assert(p.is_file())
        
        with h5py.File(file_path, 'r') as h5_file:
            _, raw_output = h5_file.items()
            self.shape = raw_output[1].shape
            
    def __getitem__(self, i):
        # get data
        with h5py.File(self.fp, 'r') as h5_file:
            f_r, r_o = h5_file.items()
            x = r_o[1][:, i * self.chunk_size : (i + 1) * self.chunk_size]
            x = torch.from_numpy(x)
        #if self.transform:
        #    x = self.transform(x)
        #else:
        #    

        # get label
            y = f_r[1][:, i * self.chunk_size : (i + 1) * self.chunk_size]
            y = torch.from_numpy(y)
        return (x, y)

    def __len__(self):
        return self.shape[1] // self.chunk_size
    

        
        
from monai import data

import torch
from torch.utils.data import Dataset

# Add project root to python path
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from neurojepa.data.transforms import loading_transforms

import argparse
import pandas as pd
from tqdm import tqdm


class MRIDatasetCache(Dataset):
    def __init__(self, roi, in_channels, csv_file, cache_dir=None):
        self.data = pd.read_csv(csv_file)
        self.load = loading_transforms(roi, spacing=(1.0, 1.0, 1.0))

        self.cache_dir = cache_dir
        self.cache_dataset = data.PersistentDataset(
            data=list([{"image": d} for d in self.data['img_path']]), 
            transform=self.load, 
            cache_dir=self.cache_dir,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            image = self.cache_dataset.__getitem__(idx)
            print(f"image: {image['image'].shape}")
            return image
        except:
            print("Error: {}".format(idx))


parser = argparse.ArgumentParser(description='Example of a command-line argument parser')

# Positional argument
parser.add_argument('--start_idx', type=int, help='Start index')

# Optional argument
parser.add_argument('--end_idx', type=int, help='End index')

parser.add_argument('--csv_file', type=str, required=True, help='Path to CSV file')
parser.add_argument('--cache_dir', type=str, required=True, help='Path to cache directory')

# Flag argument (boolean)
args = parser.parse_args()

device = torch.device("cpu")

roi = [100, 120, 100]
csv_file = args.csv_file
cache_dir = args.cache_dir

train_ds = MRIDatasetCache(
    roi, 
    in_channels=1, 
    csv_file=csv_file, 
    cache_dir=cache_dir, 
)

for idx in tqdm(range(args.start_idx, args.end_idx)):
    try:
        b = train_ds.__getitem__(idx)
    except:
        print("Error: {}".format(idx))
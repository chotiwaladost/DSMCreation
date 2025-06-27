import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import zipfile
import zlib

class NrwDataSet(Dataset):
    def __init__(self, npz_dir, amount):
        outlier_file = open("/home/julian/scripts/DSM_tiles_complete/dataset/correction/outliers_checked_stayed.txt")
        outliers = [os.path.basename(line.rstrip()) for line in outlier_file]

        c = 0
        self.dataset = []
        self.npz_dir = npz_dir
        for file in os.listdir(npz_dir):
            if not outliers.__contains__(file):

                self.dataset.append(os.path.join(npz_dir, file))

                if amount > 0:
                    c += 1
                    if c >= amount:
                        break

        outlier_file.close()

    def __len__(self):
        return len(self.dataset)

    def __getitem_by_name__(self, dataframename):
        dataframepath = os.path.join(self.npz_dir, dataframename)

        try:
            dataframe = np.load(dataframepath, allow_pickle=True)
            red = dataframe["red"]
            green = dataframe["green"]
            blue = dataframe["blue"]
            nir = dataframe["nir"]
            dom = dataframe["dom"]

            sentinel = np.stack((red, green, blue, nir))

            sentinel = torch.Tensor(sentinel)
            dsm = torch.Tensor(dom)

            return sentinel, dsm, dataframepath

        except (zipfile.BadZipFile, zlib.error, KeyError) as e:
            # Catch corrupted files OR files missing expected keys
            print(f"WARNING: Skipping corrupted or invalid file: {dataframepath}. Error: {e}")
            # Return None so the DataLoader knows this sample failed to load
            return None

    def __getitem__(self, index):
        dataframepath = self.dataset[index]
        dataframe = np.load(dataframepath, allow_pickle=True)

        red = dataframe["red"]
        green = dataframe["green"]
        blue = dataframe["blue"]
        nir = dataframe["nir"]
        dom = dataframe["dom"]

        sentinel = np.stack((red, green, blue, nir))

        sentinel = torch.Tensor(sentinel)
        dsm = torch.Tensor(dom)

        return sentinel, dsm, dataframepath

# Define a "safe" collate function that filters out None values
def collate_fn_safe(batch):
    # 'batch' is a list of results from __getitem__
    # Filter out any samples that returned None (due to loading errors)
    batch = [item for item in batch if item is not None]
    if not batch: # If the whole batch was corrupt
        # Return empty tensors or None, which your training loop should handle
        return None, None, None
    # Use the default collate function on the cleaned batch
    return torch.utils.data.dataloader.default_collate(batch)


def get_loader(npz_dir, batch_size, num_workers=2, pin_memory=True, shuffle=True, amount=0):
    train_ds = NrwDataSet(npz_dir, amount)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        collate_fn=collate_fn_safe
    )
    return train_loader


def get_dataset(npz_dir, amount):
    return NrwDataSet(npz_dir, amount=amount)

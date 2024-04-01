import os
from typing import Any, Callable, Dict, Optional
import collections

import pandas as pd
import rasterio
from PIL import Image
from torch import Tensor
from torchgeo.datasets.geo import NonGeoDataset
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from .transforms import get_sapclip_transform


CHECK_MIN_FILESIZE = 10000 # 10kb



class SAPCLIP_Dataset(NonGeoDataset):
    """S2-100K dataset.

    This dataset contains 100,000 256x256 patches of 12 band Sentinel imagery sampled randomly
    from Sentinel 2 scenes on the Microsoft Planetary Computer that have <20% cloud cover,
    intersect land, and were captured between 2021-01-01 and 2023-05-17 (there are 2,359,972
    such scenes).
    """

    validation_filenames = [
        "patch_0.tif.jpg",
        "patch_99999.tif.jpg",
    ]

    def __init__(
        self,
        root: str,
        transform_type: str = 'sapclip',
        crop_size: int = 256, 
        mode: Optional[str] = "both",
        prototype: bool=False
    ) -> None:
        """Initialize a new S2-100K dataset instance.
        Args:
            root: root directory of S2-100K pre-sampled dataset
            transform: torch transform to apply to a sample
            mode: which data to return (options are "both" or "points"), useful for embedding locations without loading images 
        """
        assert mode in ["both", "points"]
        self.root = root
        self.mode = mode
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

        index_fn = "index_bing_full.csv"

        df = pd.read_csv(os.path.join(self.root, index_fn))
        ### for prototyping
        if prototype:
            print('Using Dummy Prototype Data')
            df = df.iloc[0:100]
        self.filenames = []
        self.points = []

        n_skipped_files = 0
        for i in range(df.shape[0]):
            filename = os.path.join(self.root, df.iloc[i]["fn"])

            # if os.path.getsize(filename) < CHECK_MIN_FILESIZE:
            #     n_skipped_files += 1
            #     continue

            self.filenames.append(filename)
            self.points.append(
                (df.iloc[i]["lon"], df.iloc[i]["lat"])
            )
        self.points = torch.tensor(self.points)

        print(f"skipped {n_skipped_files}/{len(df)} images because they were smaller "
              f"than {CHECK_MIN_FILESIZE} bytes... they probably contained nodata pixels")

        if transform_type=='pretrained':
            self.transform = get_pretrained_s2_train_transform(resize_crop_size=crop_size)
        elif transform_type=='default':
            self.transform = get_s2_train_transform()
        elif transform_type=='sapclip':
            self.transform = get_sapclip_transform(resize_crop_size=crop_size)
        else:
            raise ValueError('Invalid transform type')

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.
        Args:
            index: index to return
        Returns:
            dictionary with "image" and "point" keys where point is in (lon, lat) format
        """
        point = self.points[index]
        sample = {"point": point}

        if self.mode == "both":
            # with rasterio.open(self.filenames[index]) as f:
            #     data = f.read().astype(np.float32)
            # img = torch.tensor(data)
            data = Image.open(self.filenames[index])
            data = data.convert('RGB')
            sample["image"] = data
            
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample

    def __len__(self) -> int:
        """Return the number of datapoints in the dataset.
        Returns:
            length of dataset
        """
        return len(self.filenames)

    def _check_integrity(self) -> bool:
        """Checks the integrity of the dataset structure.
        Returns:
            True if the dataset directories and split files are found, else False
        """
        
        for filename in self.validation_filenames:
            filepath = os.path.join(self.root, filename)
            if not os.path.exists(filepath):
                print(filepath +' missing' )
                return False
        return True

    def plot(
        self,
        sample: Dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.
        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle
        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = np.rollaxis(sample["image"].numpy(), 0, 3)
        ncols = 1

        fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))

        ax.imshow(image[:, :, [3,2,1]] / 4000)
        ax.axis("off")

        if show_titles:
            ax.set_title(f"({sample['point'][0]:0.4f}, {sample['point'][1]:0.4f})")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig

#manage the shape of the batch
def collate_fn(batch):
    collate_batch = {}
    collate_batch = collections.defaultdict(list)
    # collect all the dictionary into a single dictionary
    for d in batch:
        for k,v in d.items():
            collate_batch[k].append(v)

    #stack and concat to get the desired shape
    for key in collate_batch.keys():
        if key=='image':
            collate_batch[key] = torch.cat(collate_batch[key], dim=0)
        else:
            collate_batch[key] = torch.stack(collate_batch[key], dim=0)

    #define a zero matrix for label of the size (batch_size, total_number_of_images)
    loc_size = len(collate_batch['scale'])
    im_size = len(collate_batch['image'])
    loc_to_img_label = torch.zeros((loc_size,im_size))
    curr_idx=0
    #fill all the positive images for a given loc-scale pair with ones
    for i, scale in enumerate(collate_batch['scale']):
        shift = curr_idx+scale
        loc_to_img_label[i,curr_idx:shift] = 1
        curr_idx=shift

    #assign the label matrix to the collate_batch dictionary
    collate_batch['label'] = loc_to_img_label
    return collate_batch


def get_split_dataset(dataset, 
    val_split: float=0.1,
    batch_size: int=8,
     num_workers: int=4):
    train_split = 1-val_split
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_split, val_split])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
     shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
    collate_fn=collate_fn)    
    return train_loader, val_loader

if __name__ == '__main__':
    root = '/home/a.dhakal/active/project_crossviewmap/SatCLIP/sat_images_17'
    dataset = SAPCLIP_Dataset(root=root, prototype=True)
    train_loader, val_loader = get_split_dataset(dataset, 0.1, 4, 0)
    sample = next(iter(val_loader))
    import code; code.interact(local=dict(globals(),**locals()))    

    
    
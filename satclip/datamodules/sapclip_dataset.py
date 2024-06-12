import os
from typing import Any, Callable, Dict, Optional
import collections

import random
import pandas as pd
import rasterio
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torchgeo.datasets.geo import NonGeoDataset
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import h5py as h5

import lightning.pytorch as pl
from torch.utils.data import DataLoader

#relative imports 
from .transforms import get_sapclip_transform, get_none_transform



CHECK_MIN_FILESIZE = 10000 # 10kb


def get_split_dataset(dataset, 
    val_split: float=0.1,
    batch_size: int=8,
     num_workers: int=4):
    train_split = 1-val_split
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_split, val_split])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
     shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
    collate_fn=collate_fn, drop_last=True)    
    return train_loader, val_loader

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
    return dict(collate_batch)

    
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
        missing_files = ['patch_45116.tif.jpeg', 'patch_76967.tif.jpeg', 'patch_27934.tif.jpeg', 'patch_45120.tif.jpeg', 'patch_85627.tif.jpeg', 'patch_561.tif.jpeg', 'patch_381.tif.jpeg', 'patch_27139.tif.jpeg', 'patch_91793.tif.jpeg']
        corrupt_files = ['patch_151.tif.jpeg',
                        'patch_153.tif.jpeg',
                        'patch_290.tif.jpeg',
                        'patch_725.tif.jpeg',
                        'patch_932.tif.jpeg',
                        'patch_1752.tif.jpeg',
                        'patch_1920.tif.jpeg',
                        'patch_1947.tif.jpeg',
                        'patch_2188.tif.jpeg',
                        'patch_2455.tif.jpeg',
                        'patch_3110.tif.jpeg',
                        'patch_3309.tif.jpeg',
                        'patch_4814.tif.jpeg',
                        'patch_4930.tif.jpeg',
                        'patch_6994.tif.jpeg',
                        'patch_8209.tif.jpeg',
                        'patch_8544.tif.jpeg',
                        'patch_8583.tif.jpeg',
                        'patch_8899.tif.jpeg',
                        'patch_9400.tif.jpeg',
                        'patch_9514.tif.jpeg',
                        'patch_9603.tif.jpeg',
                        'patch_10110.tif.jpeg',
                        'patch_11234.tif.jpeg',
                        'patch_11878.tif.jpeg',
                        'patch_13142.tif.jpeg',
                        'patch_13707.tif.jpeg',
                        'patch_15402.tif.jpeg',
                        'patch_15590.tif.jpeg',
                        'patch_16103.tif.jpeg',
                        'patch_16167.tif.jpeg',
                        'patch_18292.tif.jpeg',
                        'patch_19122.tif.jpeg',
                        'patch_19375.tif.jpeg',
                        'patch_20964.tif.jpeg',
                        'patch_22865.tif.jpeg',
                        'patch_23379.tif.jpeg',
                        'patch_23382.tif.jpeg',
                        'patch_23666.tif.jpeg',
                        'patch_23798.tif.jpeg',
                        'patch_24247.tif.jpeg',
                        'patch_24452.tif.jpeg',
                        'patch_24911.tif.jpeg',
                        'patch_25013.tif.jpeg',
                        'patch_26207.tif.jpeg',
                        'patch_26219.tif.jpeg',
                        'patch_26555.tif.jpeg',
                        'patch_26608.tif.jpeg',
                        'patch_26713.tif.jpeg',
                        'patch_26772.tif.jpeg',
                        'patch_26800.tif.jpeg',
                        'patch_26969.tif.jpeg',
                        'patch_27263.tif.jpeg',
                        'patch_27639.tif.jpeg',
                        'patch_27843.tif.jpeg',
                        'patch_27847.tif.jpeg',
                        'patch_28078.tif.jpeg',
                        'patch_28518.tif.jpeg',
                        'patch_28530.tif.jpeg',
                        'patch_28694.tif.jpeg',
                        'patch_29134.tif.jpeg',
                        'patch_29206.tif.jpeg',
                        'patch_29262.tif.jpeg',
                        'patch_29772.tif.jpeg',
                        'patch_29949.tif.jpeg',
                        'patch_31020.tif.jpeg',
                        'patch_31254.tif.jpeg',
                        'patch_31686.tif.jpeg',
                        'patch_32220.tif.jpeg',
                        'patch_32865.tif.jpeg',
                        'patch_33151.tif.jpeg',
                        'patch_33245.tif.jpeg',
                        'patch_33364.tif.jpeg',
                        'patch_33864.tif.jpeg',
                        'patch_33946.tif.jpeg',
                        'patch_34019.tif.jpeg',
                        'patch_34079.tif.jpeg',
                        'patch_34133.tif.jpeg',
                        'patch_34333.tif.jpeg',
                        'patch_34418.tif.jpeg',
                        'patch_34520.tif.jpeg',
                        'patch_34706.tif.jpeg',
                        'patch_34768.tif.jpeg',
                        'patch_34818.tif.jpeg',
                        'patch_34855.tif.jpeg',
                        'patch_34928.tif.jpeg',
                        'patch_35183.tif.jpeg',
                        'patch_35398.tif.jpeg',
                        'patch_35475.tif.jpeg',
                        'patch_36279.tif.jpeg',
                        'patch_36951.tif.jpeg',
                        'patch_37290.tif.jpeg',
                        'patch_37393.tif.jpeg',
                        'patch_37802.tif.jpeg',
                        'patch_37815.tif.jpeg',
                        'patch_37866.tif.jpeg',
                        'patch_37921.tif.jpeg',
                        'patch_37994.tif.jpeg',
                        'patch_38136.tif.jpeg',
                        'patch_38154.tif.jpeg',
                        'patch_38497.tif.jpeg',
                        'patch_38498.tif.jpeg',
                        'patch_38545.tif.jpeg',
                        'patch_38763.tif.jpeg',
                        'patch_38868.tif.jpeg',
                        'patch_38902.tif.jpeg',
                        'patch_38960.tif.jpeg',
                        'patch_39783.tif.jpeg',
                        'patch_42560.tif.jpeg',
                        'patch_44348.tif.jpeg',
                        'patch_44384.tif.jpeg',
                        'patch_44527.tif.jpeg',
                        'patch_44670.tif.jpeg',
                        'patch_45389.tif.jpeg',
                        'patch_47889.tif.jpeg',
                        'patch_49706.tif.jpeg',
                        'patch_49965.tif.jpeg',
                        'patch_50479.tif.jpeg',
                        'patch_50518.tif.jpeg',
                        'patch_50612.tif.jpeg',
                        'patch_50789.tif.jpeg',
                        'patch_54195.tif.jpeg',
                        'patch_54795.tif.jpeg',
                        'patch_55929.tif.jpeg',
                        'patch_57059.tif.jpeg',
                        'patch_57663.tif.jpeg',
                        'patch_58174.tif.jpeg',
                        'patch_58339.tif.jpeg',
                        'patch_58758.tif.jpeg',
                        'patch_58909.tif.jpeg',
                        'patch_58959.tif.jpeg',
                        'patch_59832.tif.jpeg',
                        'patch_63584.tif.jpeg',
                        'patch_64438.tif.jpeg',
                        'patch_64488.tif.jpeg',
                        'patch_64598.tif.jpeg',
                        'patch_65389.tif.jpeg',
                        'patch_65660.tif.jpeg',
                        'patch_67047.tif.jpeg',
                        'patch_67385.tif.jpeg',
                        'patch_68721.tif.jpeg',
                        'patch_69191.tif.jpeg',
                        'patch_69841.tif.jpeg',
                        'patch_69894.tif.jpeg',
                        'patch_70129.tif.jpeg',
                        'patch_70206.tif.jpeg',
                        'patch_70444.tif.jpeg',
                        'patch_71497.tif.jpeg',
                        'patch_72464.tif.jpeg',
                        'patch_72748.tif.jpeg',
                        'patch_73500.tif.jpeg',
                        'patch_77630.tif.jpeg',
                        'patch_77663.tif.jpeg',
                        'patch_78279.tif.jpeg',
                        'patch_78463.tif.jpeg',
                        'patch_78654.tif.jpeg',
                        'patch_78973.tif.jpeg',
                        'patch_79324.tif.jpeg',
                        'patch_79743.tif.jpeg',
                        'patch_80047.tif.jpeg',
                        'patch_81183.tif.jpeg',
                        'patch_81200.tif.jpeg',
                        'patch_83182.tif.jpeg',
                        'patch_84497.tif.jpeg',
                        'patch_84883.tif.jpeg',
                        'patch_85142.tif.jpeg',
                        'patch_85561.tif.jpeg',
                        'patch_85652.tif.jpeg',
                        'patch_86482.tif.jpeg',
                        'patch_87912.tif.jpeg',
                        'patch_89474.tif.jpeg',
                        'patch_90753.tif.jpeg',
                        'patch_91095.tif.jpeg',
                        'patch_92411.tif.jpeg',
                        'patch_92604.tif.jpeg',
                        'patch_93177.tif.jpeg',
                        'patch_93565.tif.jpeg',
                        'patch_94455.tif.jpeg',
                        'patch_96027.tif.jpeg',
                        'patch_96237.tif.jpeg',
                        'patch_98965.tif.jpeg',
                        'patch_99366.tif.jpeg']
        
        bad_files = corrupt_files + missing_files
        self.root = root
        self.mode = mode
        # if not self._check_integrity():
        #     raise RuntimeError("Dataset not found or corrupted.")

        index_fn = "index_sentinel_full.csv"
        #add full path of corrupt files
        corrupt_files = [os.path.join(self.root, 'sentinel', file) for file in bad_files]

        df = pd.read_csv(os.path.join(self.root, index_fn))
        ### for prototyping
        if prototype:
            print('Using Dummy Prototype Data')
            df = df.iloc[0:100]
        self.filenames = []
        self.points = []

        n_skipped_files = 0
        for i in range(df.shape[0]):
            filename = os.path.join(self.root, 'sentinel', df.iloc[i]["fn"])
            if filename in corrupt_files:
                n_skipped_files+=1
                continue        

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
            print('No transform used')
            self.transform=None


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
            try:
                data = Image.open(self.filenames[index])
            # if unable to open file
            except UnidentifiedImageError as e:
                print('UnidentifiedImageError')
                index = random.randint(0,1000)
                data = Image.open(self.filenames[index])
                point = self.points[index]
                sample = {'point':point}
            
            data = data.convert('RGB')
            sample["image"] = data

        if self.transform is not None:
            sample = self.transform(sample)
        else:
        
            sample['image'] = np.array(data)
            sample['point'] = point.numpy()
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

class SAPCLIP_Dataset_H5(torch.utils.data.Dataset):
    def __init__(
        self,
        input_path: str,
        transform_type: str = 'sapclip',
        crop_size: int = 256, 
    ):
        self.h5_file = h5.File(input_path, 'r')
        self.images = self.h5_file['images']
        self.points = self.h5_file['lon_lat']
        
        #define the trasnform type
        if transform_type=='pretrained':
            self.transform = get_pretrained_s2_train_transform(resize_crop_size=crop_size)
        elif transform_type=='default':
            self.transform = get_s2_train_transform()
        elif transform_type=='sapclip':
            self.transform = get_sapclip_transform(resize_crop_size=crop_size)
        else:
            print('No transform used')
            self.transform=None

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx]).convert('RGB')
        point = self.points[idx]
        sample={}
        sample['image'] = image
        sample['point'] = torch.tensor(point).double()

        #apply transformation
        if self.transform is not None:
            sample = self.transform(sample)
        else:
            sample=sample
        
        return sample
        


if __name__ == '__main__':
    # 

    # dataset = SAPCLIP_Dataset(root=root, prototype=False)
    # train_loader, val_loader = get_split_dataset(dataset, 0.1, 512, 8)
    # sample = next(iter(val_loader))
    # i=0
    # for s in train_loader:
    #     batch = s
    #     print(f'Sample {i}')
    #     i+=1
    data_type='h5'
    if data_type=='h5':
        print('H5')
        path = '/scratch/a.dhakal/hyper_satclip/data/h5_data/satclip_data.h5'
        dataset = SAPCLIP_Dataset_H5(input_path=path, transform_type='sapclip')
    elif data_type=='normal':
        print('Normal')
        path = '/scratch/a.dhakal/hyper_satclip/data/satclip_data/satclip_sentinel/images'
        dataset = SAPCLIP_Dataset(root=path, transform_type='sapclip', crop_size=224, prototype=False)
    # handle = h5.File(h5_path, 'r')
    # images = handle['images']
    # points = handle['lon_lat']
    
    train_loader, val_loader = get_split_dataset(dataset, val_split=0.05, batch_size=512,
     num_workers=8)
    
    from tqdm import tqdm
    for batch in tqdm(train_loader):
        b = batch
    import code; code.interact(local=dict(globals(), **locals()))
    
    
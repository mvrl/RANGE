## this script is used to generate the database for the range model

import torch
import numpy as np
import argparse
from huggingface_hub import hf_hub_download
from typing import Any, Callable, Dict, Optional
from torch import Tensor
import pandas as pd
import os
import rasterio
from tqdm import tqdm
from PIL import Image
#local import 
from .datamodules.transforms import get_rgb_val_transform, get_multi_spec_val_transform
from .load import get_satclip
from .datamodules.s2geo_dataset import S2Geo
from .vision_models.satmae import SatMAE_Raw

def get_args():
    parser = argparse.ArgumentParser(description='Create a database of embeddings')
    parser.add_argument('--out_path', type=str, help='Path to the save output', default='/home/a.dhakal/active/user_a.dhakal/hyper_satclip/data/data/models/ranf/ranf_satmae_db.npz')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--to_do', type=str, default='make_db', choices=['make_db', 'eval'], help='What to do')
    #dataloader args
    parser.add_argument('--data_dir', type=str, default='/home/a.dhakal/active/project_crossviewmap/SatCLIP', help='Path to the data')
    parser.add_argument('--rgb_path', type=str, default='/home/a.dhakal/active/project_crossviewmap/SatCLIP/rgb_sentinel')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--inp_chan', type=int, default=12, help='Number of input channels')
    parser.add_argument('--visual_encoder', type=str, choices=['SatMAE', 'Other'], default='SatMAE', help='Number of output channels')
    args = parser.parse_args()
    return args

CHECK_MIN_FILESIZE = 10000 # 10kb
class SATCLIP_VALDS(torch.utils.data.Dataset):

    """S2-100K dataset.

    This dataset contains 100,000 256x256 patches of 12 band Sentinel imagery sampled randomly
    from Sentinel 2 scenes on the Microsoft Planetary Computer that have <20% cloud cover,
    intersect land, and were captured between 2021-01-01 and 2023-05-17 (there are 2,359,972
    such scenes).
    """

    validation_filenames = [
        "index.csv",
        "images/",
        "images/patch_0.tif",
        "images/patch_99999.tif",
    ]

    def __init__(
        self,
        root: str,
        rgb_path = None,
        transform: str='rgb',
        mode:str = "rgb",
        crop_size:int = 224,
        propotype=False
    ) -> None:
        """Initialize a new S2-100K dataset instance.
        Args:
            root: root directory of S2-100K pre-sampled dataset
            transform: torch transform to apply to a sample
            mode: whether the input data is rgb or multispectral
        """
        assert mode in ["rgb", "multispec","both"]
        self.root = root
        self.transform = transform
        self.mode = mode
        # if not self._check_integrity():
        #     raise RuntimeError("Dataset not found or corrupted.")

        index_fn = "index.csv"

        df = pd.read_csv(os.path.join(self.root, index_fn))
        self.filenames = []
        self.points = []

        n_skipped_files = 0
        unavailable_files = 0
        #get all the files from original sentinel 12 bands
        existing_old_filenames = os.listdir(os.path.join(self.root, "images"))
        #get all the files from the rgb
        existing_rgb_filenames = [''.join((file.split('.')[:-1])) for file in os.listdir(rgb_path)]
        existing_rgb_filenames = [f+'.tif' for f in existing_rgb_filenames]
        #only keep files that are common to both
        existing_filenames = list(set(existing_old_filenames) & set(existing_rgb_filenames))
        existing_filenames = [os.path.join(self.root, "images", fn) for fn in existing_filenames]
        
        for i in range(df.shape[0]):
            filename = os.path.join(self.root, "images", df.iloc[i]["fn"])
            if filename not in existing_filenames:
                unavailable_files += 1
                continue
            if os.path.getsize(filename) < CHECK_MIN_FILESIZE:
                n_skipped_files += 1
                continue

            self.filenames.append(filename)
            self.points.append(
                (df.iloc[i]["lon"], df.iloc[i]["lat"])
            )
        self.points = torch.tensor(self.points)
        #
        print(f"skipped {n_skipped_files}/{len(df)} images because they were smaller "
              f"than {CHECK_MIN_FILESIZE} bytes... they probably contained nodata pixels")
        print(f"skipped {unavailable_files} images because they were not found in the images directory")
        ##get the filenames for the rgb path
        patch_names = [patch.split('/')[-1].replace('.tif','.jpg') for patch in existing_filenames]
        self.rgb_filenames = [os.path.join(rgb_path, patch_name) for patch_name in patch_names]
        #filter files that do not exist
        #get the transform
        # if transform=='pretrained':
        #     self.train_transform = get_pretrained_s2_train_transform(resize_crop_size=crop_size)
        # elif transform=='default':
        #     self.train_transform = get_s2_train_transform()
        # elif transform=='rgb':
        #     print('Using RGB transform.')
        #     self.train_transform = get_rgb_val_transform(resize_crop_size=crop_size)
        # else:
        #     self.train_transform = transform
        self.rgb_transform = get_rgb_val_transform(resize_crop_size=crop_size)
        self.multi_spec_transform = get_multi_spec_val_transform(resize_crop_size=crop_size)
        


    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.
        Args:
            index: index to return
        Returns:
            dictionary with "image" and "point" keys where point is in (lon, lat) format
        """
        point = self.points[index]
        sample_original = {"point": point}
        sample_new = {'point': point}
         
        if self.mode == "multi_spec":
            with rasterio.open(self.filenames[index]) as f:
                data = f.read().astype(np.float32)
        elif self.mode == "rgb":
             data = Image.open(self.filenames[index])
             data.convert('RGB')
        elif self.mode == "both":
            #first process the multispectral data
            with rasterio.open(self.filenames[index]) as f:
                data = f.read().astype(np.float32)
            sample_original['image'] = data
#           now process the rgb data
            rgb_data = Image.open(self.rgb_filenames[index])
            rgb_data.convert('RGB')
            sample_new['image'] = rgb_data
            
            #trasnform the data
            if self.transform:
                sample_original = self.multi_spec_transform(sample_original)
                sample_new = self.rgb_transform(sample_new)
        

        sample = {'point': point, 'image_original': sample_original['image'],
         'image_new': sample_new['image']}
            
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


def create_database(image_model, satclip_model, dataloader, out_path, device='cuda'):
    #double check and set to eval mode
    image_model.eval()
    satclip_model.eval()
    image_embeddings_list = []
    satclip_embeddings_list = []
    loc_list = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader)):
            image_original = data['image_original'].to(device).double()
            image_new = data['image_new'].to(device).double()
            loc = data['point'].double()
            #compute the location and image embeddings
            satclip_embeddings = satclip_model(image_original).detach().cpu().numpy()
            image_embeddings = image_model(image_new).detach().cpu().numpy()
            #save the embeddings and location
            satclip_embeddings_list.append(satclip_embeddings)
            image_embeddings_list.append(image_embeddings)
            
            loc_list.append(loc)
    #save the embeddings as npz file
    all_image_embeddings = np.concatenate(image_embeddings_list, axis=0)
    all_satclip_embeddings = np.concatenate(satclip_embeddings_list, axis=0)
    all_loc_list = np.concatenate(loc_list, axis=0)
    np.savez(out_path, locs=all_loc_list,
        image_embeddings=all_image_embeddings,
        satclip_embeddings=all_satclip_embeddings)
    print(f'Database created and saved to {out_path}')

    
if __name__ == '__main__':
    args = get_args()
    #get the satclip model

    #create the dataset
    dataset = SATCLIP_VALDS(root=args.data_dir,
    transform=True,
    mode='both',
    rgb_path=args.rgb_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
    shuffle=False, num_workers=args.num_workers, drop_last=False)
    #grab the image and location models
    image_model = SatMAE_Raw().eval().to(args.device).double()
    location_model = get_satclip(
            hf_hub_download("microsoft/SatCLIP-ViT16-L40", "satclip-vit16-l40.ckpt", force_download=False),
                device = args.device, return_all=True).eval().double()
    satclip_image_model = location_model.visual.eval()
    
    #create the database
    if args.to_do == 'make_db':
        create_database(image_model=image_model, satclip_model=satclip_image_model, 
        dataloader=dataloader, out_path=args.out_path, device=args.device)
    else:
        raise ValueError('Invalid option for to_do. Must be make_db')

import torch
import numpy as np
import argparse
from huggingface_hub import hf_hub_download
from typing import Any, Callable, Dict, Optional
from torch import Tensor
import pandas as pd
import os
from PIL import Image
#local import 
from .datamodules.transforms import get_rgb_val_transform
from .load import get_satclip
from .datamodules.s2geo_dataset import S2Geo
from .vision_models.satmae import SatMAE_Raw

def get_args():
    parser = argparse.ArgumentParser(description='Create a database of embeddings')
    parser.add_argument('--out_dir', type=str, help='Path to the checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--to_do', type=str, default='make_db', choices=['make_db', 'eval'], help='What to do')
    #dataloader args
    parser.add_argument('--data_dir', type=str, default='/projects/bdec/adhakal2/hyper_satclip/data/original_satclip', help='Path to the data')
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
        transform: str='rgb',
        mode:str = "rgb",
        crop_size:int = 224,
    ) -> None:
        """Initialize a new S2-100K dataset instance.
        Args:
            root: root directory of S2-100K pre-sampled dataset
            transform: torch transform to apply to a sample
            mode: whether the input data is rgb or multispectral
        """
        assert mode in ["rgb", "multispec"]
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
        existing_filenames = os.listdir(os.path.join(self.root, "images"))
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
        print(f"skipped {n_skipped_files}/{len(df)} images because they were smaller "
              f"than {CHECK_MIN_FILESIZE} bytes... they probably contained nodata pixels")
        print(f"skipped {unavailable_files} images because they were not found in the images directory")
        #filter files that do not exist
        #get the transform
        if transform=='pretrained':
            self.train_transform = get_pretrained_s2_train_transform(resize_crop_size=crop_size)
        elif transform=='default':
            self.train_transform = get_s2_train_transform()
        elif transform=='rgb':
            print('Using RGB transform.')
            self.train_transform = get_rgb_val_transform(resize_crop_size=crop_size)
        else:
            self.train_transform = transform
        


    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.
        Args:
            index: index to return
        Returns:
            dictionary with "image" and "point" keys where point is in (lon, lat) format
        """
        point = torch.tensor(self.points[index])
        sample = {"point": point}
       
         
        if self.mode == "multi_spec":
            with rasterio.open(self.filenames[index]) as f:
                data = f.read().astype(np.float32)
        elif self.mode == "rgb":
             data = Image.open(self.filenames[index])
             data.convert('RGB')

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


def create_database(model, dataloader, outpath, device='cuda'):
    location_encoder = model.location.eval()
    image_encoder = model.visual
    import code; code.interact(local=dict(globals(), **locals()))
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            image = data['image'].to(device)
            y = data['point']
            #get the embeddings
            image_features = model(x, y)
            embeddings = embeddings.cpu().numpy()
            #save the embeddings
            np.savez(outpath + f'/embeddings_{i}.npz', embeddings=embeddings)
    
if __name__ == '__main__':
    args = get_args()
    #get the satclip model
    # model = get_satclip(
    #         hf_hub_download("microsoft/SatCLIP-ViT16-L40", "satclip-vit16-l40.ckpt", force_download=False),
    #             device = args.device, return_all=True)
    
    #get the dataset
    # datamodule = S2GeoDataModule(data_dir=args.data_dir,
    #     batch_size=args.batch_size, num_workers=args.num_workers, val_random_split_fraction=0.1)
    # datamodule.setup()
    # dataloader = datamodule.train_dataloader()
    dataset = SATCLIP_VALDS(root=args.data_dir,
    transform='rgb',
    mode='rgb')
    import code; code.interact(local=dict(globals(), **locals()))
    #create the database
    if args.to_do == 'make_db':
        create_database(model, dataloader, args.out_dir)

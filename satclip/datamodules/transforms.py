import torchvision.transforms as T
import torch
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform  
from albumentations.pytorch import ToTensorV2
import numpy as np


def get_train_transform(resize_crop_size = 256,
                  mean = [0.4139, 0.4341, 0.3482, 0.5263],
                  std = [0.0010, 0.0010, 0.0013, 0.0013]
                  ):

    augmentation = A.Compose(
        [
            A.RandomResizedCrop(height=resize_crop_size, width=resize_crop_size),
            A.RandomBrightnessContrast(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.GaussianBlur(),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    def transform(sample):
        image = sample["image"].numpy().transpose(1,2,0)
        point = sample["point"]

        image = augmentation(image=image)["image"]
        point = coordinate_jitter(point)

        return dict(image=image, point=point)

    return transform

def get_s2_train_transform(resize_crop_size = 256):
    augmentation = T.Compose([
        T.RandomCrop(resize_crop_size),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.GaussianBlur(3),
    ])

    def transform(sample):
        image = sample["image"] / 10000.0
        point = sample["point"]
        image = torch.tensor(image)
        image = augmentation(image)
        point = coordinate_jitter(point)
        return dict(image=image, point=point)

    return transform

def get_pretrained_s2_train_transform(resize_crop_size = 256):
    augmentation = T.Compose([
        T.RandomCrop(resize_crop_size),
        T.RandomVerticalFlip(),
        T.RandomHorizontalFlip(),
        T.GaussianBlur(3),
    ])

    def transform(sample):
        image = sample["image"] / 10000.0
        point = sample["point"]

        B10 = np.zeros((1, *image.shape[1:]), dtype=image.dtype)
        image = np.concatenate([image[:10], B10, image[10:]], axis=0)
        image = torch.tensor(image)

        image = augmentation(image)

        point = coordinate_jitter(point)

        return dict(image=image, point=point)

    return transform

def get_sapclip_transform(resize_crop_size=256):
    augmentation = T.Compose([
        T.RandomCrop(resize_crop_size),
        T.RandomVerticalFlip(),
        T.RandomHorizontalFlip(),
        T.GaussianBlur(3),
        T.ToTensor()
    ])

    def transform(sample):
        image = sample['image']
        point = sample['point']
        
        #define the maximum number of crops possible at the largest scale
        max_crops = 5
        #randomly select the scale
        scale = np.random.choice([1, 3, 5])
        #create the bounding box for the scale
        crop_size = 256*scale
        #create a mask of ones 
        mask = torch.zeros((3,256,256))
        #center crop image for the given scale
        image = T.CenterCrop(size=crop_size)(image)

        #create crops for the given scale
        num_crops = scale 
        multi_images = [T.RandomCrop(256)(image) for i in range(num_crops)]
        multi_images = torch.stack([augmentation(img) for img in multi_images], dim=0)
        #jitter the point
        point = coordinate_jitter(point)
########################################### Comment for now ####################################
        # #compute the number of masks to add ##### hardcoded need to make dynamic
        # num_masks_to_add = max_crops-num_crops ##currently takes scale as the number of crops
        # [multi_images.append(mask) for i in range(num_masks_to_add)]
        # multi_images = torch.stack(multi_images, dim=0)
        # #calculate the valid mask
        # valid_mask = torch.tensor([1]*num_crops + [0]*(max_crops-num_crops))
        
        #repeat everything else to fit the max crop
        # point = point.repeat(max_crops,1)
        # scale = torch.tensor([scale]*max_crops)
########################################### Comment for now ####################################
        return dict(image=multi_images, point=point, scale=torch.tensor(scale))
    
    return transform 


def coordinate_jitter(
        point,
        radius=0.01 # approximately 1 km
    ):
    return point + torch.rand(point.shape) * radius


import torchvision.transforms as T
import torch
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform  
from albumentations.pytorch import ToTensorV2
# from rtdl_num_embeddings import PiecewiseLinearEncoding
import numpy as np
import torch.nn as nn


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


def get_none_transform():
    def transform(sample):
        image = sample['image'].numpy()
        point = sample['point']
        return dict(image=image, point=point)

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
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
    ])

    map_scale = {1:torch.tensor([1,0,0]), 3:torch.tensor([0,1,0]), 5:torch.tensor([0,0,1])}

    def transform(sample):
        image = sample['image']
        point = sample['point']
        
        # #define the maximum number of crops possible at the largest scale
        # max_crops = 5

        #randomly select the scale
        scale = np.random.choice([1, 3, 5])

        #create the bounding box for the scale
        crop_size = 256*scale

        #center crop image for the given scale
        image = T.CenterCrop(size=crop_size)(image)

        #create crops for the given scale
        num_crops = scale 
        multi_images = [T.RandomCrop(256)(image) for i in range(num_crops)]
        multi_images = torch.stack([augmentation(img) for img in multi_images], dim=0)
        #jitter the point
        point = coordinate_jitter(point)
        #one hot encode the scale
        one_hot_scale = map_scale[scale]
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
        return dict(image=multi_images, point=point, scale=torch.tensor(scale), hot_scale=one_hot_scale)    
    return transform 

def get_rgb_val_transform(resize_crop_size=256):
    augmentation = T.Compose([
        T.CenterCrop(resize_crop_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])

    def transform(sample):
        image = sample['image']
        point = sample['point']
        image = augmentation(image)
        return dict(image=image, point=point)
    return transform

def get_multi_spec_val_transform(resize_crop_size=256):
    augmentation = T.Compose([
        T.CenterCrop(resize_crop_size)
    ])
    def transform(sample):
        image = sample["image"] / 10000.0
        point = sample["point"]

        B10 = np.zeros((1, *image.shape[1:]), dtype=image.dtype)
        image = np.concatenate([image[:10], B10, image[10:]], axis=0)
        
        image = torch.tensor(image)
        image = augmentation(image)

        point = point

        return dict(image=image, point=point)

    return transform

#get a single crop for each sample irrespective of scale
def get_sapclip_uni_transform(resize_crop_size=256,scale_encoding='onehot', scale_bins=3, 
            scale_ratio=[1/3,1/3,1/3], crop_type='resized'):
    augmentation = T.Compose([
        T.RandomCrop(resize_crop_size),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        T.RandomVerticalFlip(),
        T.RandomHorizontalFlip(),
        T.GaussianBlur(3),
        T.ToTensor()
    ])
    crop_type = crop_type
    if crop_type == 'resized':
        print('Getting sammples for different scales by resizing')
    elif crop_type == 'sampled':
        print('Getting samples for different scales by sampling')
    else:
        raise ValueError('Invalid crop type')

    if scale_encoding == 'onehot':
        map_scale = {1:torch.tensor([1,0,0]), 3:torch.tensor([0,1,0]), 5:torch.tensor([0,0,1])}
    elif scale_encoding == 'ple':
        bins =  [torch.from_numpy(np.linspace(0,6,scale_bins+1)).double()]
        PLE = PiecewiseLinearEncoding(bins)
        scale_1_encoding = PLE(torch.tensor([1]))
        scale_3_encoding = PLE(torch.tensor([3]))
        scale_5_encoding = PLE(torch.tensor([5]))
        map_scale = {1:scale_1_encoding, 3:scale_3_encoding, 5:scale_5_encoding}
    elif scale_encoding == 'learnable':
        #scale_embeddings = nn.Embedding(3,scale_bins)
        map_scale = {1:torch.tensor(0), 3:torch.tensor(1), 5:torch.tensor(2)}

    def transform(sample):
        image = sample['image']
        point = sample['point']
        # define the different scales
        scale = np.random.choice([1, 3, 5], p=scale_ratio)

        #create the bounding box for the scale
        crop_size = 256*scale

        #center crop image for the given scale
        big_image = T.CenterCrop(size=crop_size)(image)

        #create crops for the given scale
        if crop_type == 'sampled':
            cropped_image = T.RandomCrop(256)(big_image)
        elif crop_type == 'resized':
            cropped_image = T.Resize(256)(big_image)
        cropped_image = augmentation(cropped_image)
        #jitter the point
        point = coordinate_jitter(point)
        #one hot encode the scale
        one_hot_scale = map_scale[scale]
        return dict(image=cropped_image, point=point, scale=torch.tensor(scale), hot_scale=one_hot_scale, label=torch.zeros(3))    
    return transform 

def coordinate_jitter(
        point,
        radius=0.01 # approximately 1 km
    ):
    return point + torch.rand(point.shape) * radius


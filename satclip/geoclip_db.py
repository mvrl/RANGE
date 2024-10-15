import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from geoclip import LocationEncoder, ImageEncoder
from transformers import CLIPModel, AutoProcessor
import pandas as pd
from PIL import Image

def filter_csv(csv_path, img_path):
    #load the original csv
    df = pd.read_csv(csv_path)
    #filter the data for only images that are downloaded
    downloaded_images = os.listdir(img_path)
    downloaded_images = [image.replace('-','/') for image in downloaded_images]
    filtered_df = df[df['IMG_ID'].isin(downloaded_images)]
    return filtered_df

#create the appropriate dataset
class GeoCLIPData(Dataset):
    def __init__(self, csv_path, img_path):
        self.df = filter_csv(csv_path, img_path)
        image_paths = self.df['IMG_ID'].values
        image_paths = [image.replace('/','-') for image in image_paths]
        self.image_paths = [os.path.join(img_path, image) for image in image_paths]
        self.locs = self.df[['LAT', 'LON']].values
        self.bin_id = self.df['bin_id'].values
        #define the image processor
        self.image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        #preprocess the image
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = self.image_processor(images=img, return_tensors="pt")['pixel_values'][0]
        loc = torch.tensor(self.locs[idx])
        bin_id = self.bin_id[idx]        
        return img, loc, bin_id
    
    def __len__(self):
        return len(self.image_paths)

#Create the embeddings for GeoCLIP
class GEOCLIP_FORWARD(nn.Module):
    def __init__(self):
        super(GEOCLIP_FORWARD, self).__init__()
        self.location_encoder = LocationEncoder()
        self.geoclip_image_encoder = ImageEncoder()
        self.CLIP_image_encoder = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

    def forward(self, img):
        high_res_embeddings = self.CLIP_image_encoder.get_image_features(pixel_values=img)
        low_res_embeddings = self.geoclip_image_encoder(img)
        return high_res_embeddings, low_res_embeddings, 


if __name__ == '__main__':
    #load the original csv
    csv_path = '/projects/bdec/adhakal2/hyper_satclip/data/geoclip_data/mp16_healpix_32.csv'
    img_path = '/projects/bdec/adhakal2/hyper_satclip/data/geoclip_data/MP-16/images'
    batch_size = 400
    device = 'cuda'
    dataset = GeoCLIPData(csv_path, img_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    model = GEOCLIP_FORWARD()
    model = model.to(device)
    model = model.eval()
    #define empty lists to store the embeddings
    list_high_res_embeddings = []
    list_low_res_embeddings = []
    list_locs = []
    list_bin_ids = []
    #run the forward pass
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader)):
            img, loc, bin = data
            img = img.to(device)
            high_res_embeddings, low_res_embeddings = model(img)
            list_high_res_embeddings.append(high_res_embeddings.cpu().numpy())
            list_low_res_embeddings.append(low_res_embeddings.cpu().numpy())
            list_bin_ids.append(bin)
            list_locs.append(loc.numpy())

    #convert the lists to numpy arrays
    all_high_res = np.vstack(list_high_res_embeddings)
    all_low_res = np.vstack(list_low_res_embeddings)
    #flip the loc array to lon/lat
    all_locs = np.vstack(list_locs)
    all_locs = all_locs[:,[1,0]]
    # all_bin_ids = np.vstack(np.reshape(list_bin_ids, (-1,1)))
    #save as npz file
    np.savez('/projects/bdec/adhakal2/hyper_satclip/data/models/ranf_geoclip/ranf_geoclip_db.npz', 
                image_embeddings=all_high_res, geoclip_embeddings=all_low_res, locs=all_locs)

        

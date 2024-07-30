import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
# import cartopy.crs as ccrs
import numpy as np
import math
import torch
from tqdm import tqdm
from sklearn import decomposition
from skimage import exposure
import pandas as pd

#local imports 
from .fit import SAPCLIP_PCME
#US coordinates
def get_grid(lat_start=24.520416, lat_end=49.383296, lon_start=-124.762779, lon_end=-66.948890, distance=0.02):
    lon = np.arange(lon_start, lon_end, distance)
    lat = np.arange(lat_start, lat_end, distance)
    lons, lats = np.meshgrid(lon, lat)
    grid = np.c_[lons.flatten(), lats.flatten()]
    return grid

class GridDataset(torch.utils.data.Dataset):
    def __init__(self, grid, scale):
        self.grid=grid
        self.map_scale = {1:torch.tensor([1,0,0]), 3:torch.tensor([0,1,0]), 5:torch.tensor([0,0,1])}
        self.scale = self.map_scale[scale]
    def __len__(self):
        return len(self.grid)
    def __getitem__(self, i):
        return (self.grid[i], self.scale)

if __name__=='__main__':
    scale = 3
    #load ckpt and load pretrained model
    ckpt_path='/scratch/a.dhakal/hyper_satclip/logs/SAPCLIP/pcme_models/epoch=68-val_loss=4426.006.ckpt'
    ckpt = torch.load(ckpt_path)
    model = SAPCLIP_PCME(embed_dim=256, loss_type='pcme', anneal_T=1000)
    unused_keys = model.load_state_dict(ckpt['state_dict'])
    print(unused_keys)
    
    #generate the grid
    start_lat = 41.33528310222733, 
    start_lon = -18.096465195594874
    end_lat = 64.82541458483378,
    end_lon = 64.16915646922804
    grid = get_grid()

    #get the encoder
    sapclip_encoder = model.model.eval()
    for param in sapclip_encoder.parameters():
        param.requires_grad = False
    
    #send encoder to cuda
    sapclip_encoder = sapclip_encoder.to('cuda')
    
    #prepare the input data
    grid = torch.from_numpy(grid)
    
    grid_dataset = GridDataset(grid, scale)
    grid_dataloader = torch.utils.data.DataLoader(grid_dataset, batch_size=500000, num_workers=4, shuffle=False)

    total_size = len(grid)
    coordinates = torch.zeros(total_size, 2)
    mu_embeddings = torch.zeros(total_size, 256)
    sigma_embeddings = torch.zeros(total_size, 256)
    sigma_l1 = torch.zeros(total_size,1)
        
    curr=0
    for i, data in tqdm(enumerate(grid_dataloader)):
        loc, scale = data
        curr_batch = len(scale)
        loc = loc.to('cuda')
        scale = scale.to('cuda')
        loc_mu, loc_logsigma = sapclip_encoder.encode_location(coords=loc, scale=scale)
        loc_sigma = torch.exp(loc_logsigma)
        coordinates[curr:curr+curr_batch] = loc
        mu_embeddings[curr:curr+curr_batch] = loc_mu
        sigma_embeddings[curr:curr+curr_batch] = loc_sigma
        sigma_l1_curr = loc_sigma.norm(p=1, dim=-1, keepdim=True)
        sigma_l1[curr:curr+curr_batch] = sigma_l1_curr
        curr = curr+curr_batch
    
    feats = mu_embeddings
    f_mu = feats.mean(0)
    f_std = feats.std(0)
    feats = feats - f_mu
    feats = feats / f_std

    num_ds_dims = 3
    seed = 56
    
    print('Performing dimensionality reduction.')
    dsf = decomposition.FastICA(n_components=num_ds_dims, random_state=seed, whiten='unit-variance', max_iter=1000)
    dsf.fit(feats)

    feats_ds = dsf.transform(feats)

        # equalize - doing this means there is no need to do the mean normalization
    # for cc in range(num_ds_dims):
    #     feats_ds[:, cc] = exposure.equalize_hist(feats_ds[:, cc])

    p = pd.DataFrame(feats_ds, columns=['r', 'g', 'b'])
    p['lon'] = coordinates[:,0].numpy()
    p['lat'] = coordinates[:,1].numpy()
    p['sigma'] = sigma_l1
    p.to_csv(f'/scratch/a.dhakal/hyper_satclip/random/sapclip_ica_scale_3_noequalize.csv', index=False)

    import code; code.interact(local=dict(globals(), **locals()))


        




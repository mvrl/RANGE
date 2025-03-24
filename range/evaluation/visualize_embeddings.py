"""
Extracts features from a trained network for each geo location, performs 
dimensionality reduction, and generates an output image.

Different seed values will result in different mappings of locations to colors. 
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
# import datasets
import matplotlib.pyplot as plt
import os
from sklearn import decomposition
from skimage import exposure
import json
# import utils
# import models
import argparse
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm import tqdm
import time
import sys
#local import 
from .range import LocationEncoder
from .evaldatasets import CSVDataset

def coord_grid(grid_size, split_ids=None, split_of_interest=None):
    # generate a grid of locations spaced evenly in coordinate space

    feats = np.zeros((grid_size[0], grid_size[1], 2), dtype=np.float32)
    mg = np.meshgrid(np.linspace(-180, 180, feats.shape[1]), np.linspace(90, -90, feats.shape[0]))
    feats[:, :, 0] = mg[0]
    feats[:, :, 1] = mg[1]
    if split_ids is None or split_of_interest is None:
        # return feats for all locations
        # this will be an N x 2 array
        return feats.reshape(feats.shape[0]*feats.shape[1], 2)
    else:
        # only select a subset of locations
        ind_y, ind_x = np.where(split_ids==split_of_interest)

        # these will be N_subset x 2 in size
        return feats[ind_y, ind_x, :]

def get_args():
    parser = argparse.ArgumentParser(description='Visualize embeddings')
    parser.add_argument('--location_model_name', type=str, default='RANF_HILO_softmax_15', help='Name of the location model')
    parser.add_argument('--input_csv', type=str, default='/projects/bdec/adhakal2/hyper_satclip/data/eval_data/usa_latlon.csv')
    parser.add_argument('--ranf_model', type=str, default='SatCLIP')
    parser.add_argument('--ranf_db', type=str, default='/projects/bdec/adhakal2/hyper_satclip/data/models/ranf/ranf_satmae_db.npz')
    parser.add_argument('--beta', type=float, default=0.5)
    return parser.parse_args()
# params - specify model of interest here

if __name__ == '__main__':
    args = get_args()
    args.k=1
    seed = 2001
    # with open('paths.json', 'r') as f:
    #     paths = json.load(f)
    num_ds_dims = 3
    # model_path = 'pretrained_models/model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt' 
    if 'RANF_COMBINED' in args.location_model_name:
        op_file_name = f'/projects/bdec/adhakal2/hyper_satclip/data/eval_data/eval_results/images/usa/{args.location_model_name}-beta{args.beta}'+ '_ica.png'

    else:
        op_file_name = f'/projects/bdec/adhakal2/hyper_satclip/data/eval_data/eval_results/images/usa/{args.location_model_name}'+ '_ica.png'

    # args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device='cuda'

    dataset = CSVDataset(args.input_csv)
    dataloader = DataLoader(dataset, batch_size=100000, shuffle=False, num_workers=4)
    
    model = LocationEncoder(args)
    model = model.to(args.device)
    model.eval()

    # # # load ocean mask
    # # mask = np.load('/projects/bdec/adhakal2/hyper_satclip/data/eval_data/ocean_mask.npy')
    # # mask_inds = np.where(mask.reshape(-1) == 1)[0]
    
    # locs = coord_grid(mask.shape)
    # locs = locs[mask_inds, :]
    # #this is in the form lon, lat
    # locs = torch.from_numpy(locs).double()
    # locs = locs.to(args.device)
    # batch_size = len(locs)//10000
    # # import code; code.interact(local=dict(globals(), **locals()))
    # with torch.no_grad():
    #     feats = model(locs,scale=None)

    # try:
    #     feats = feats.cpu().numpy()
    # except:
    #     feats = feats
    embeddings_list = []
    locs_list = []
    tick = time.time()
    with torch.no_grad():
        for data in tqdm(dataloader):
            locs, _, _ = data
            locs_unaltered = locs.clone()
            locs = locs.to(args.device)
            feats = model(locs,scale=None)
            try:
                feats = feats.cpu().numpy()
                locs = locs.cpu().numpy()
            except:
                feats = feats        
            embeddings_list.append(feats)
            locs_list.append(locs_unaltered)
    tock = time.time()
    print(f'Time taken to extract embeddings: {tock-tick}')
    sys.exit()
    feats = np.concatenate(embeddings_list, axis=0)
    all_locs = np.concatenate(locs_list, axis=0)
    # standardize the features
    f_mu = feats.mean(0)
    f_std = feats.std(0)
    f_std[f_std == 0] = 1e-8
    feats = feats - f_mu
    feats = feats / f_std
    # import code; code.interact(local=dict(globals(), **locals()))
    assert not np.any(np.isnan(feats))
    assert not np.any(np.isinf(feats))

    # downsample features - choose middle time step
    print('Performing dimensionality reduction.')
    dsf = decomposition.FastICA(n_components=num_ds_dims, random_state=seed, whiten='unit-variance', max_iter=1000)
    dsf.fit(feats)

    feats_ds = dsf.transform(feats)

    # equalize - doing this means there is no need to do the mean normalization
    for cc in range(num_ds_dims):
        feats_ds[:, cc] = exposure.equalize_hist(feats_ds[:, cc])
    # Create a map with Cartopy
    # Extract longitude and latitude
    longitudes = all_locs[:, 0]
    latitudes = all_locs[:, 1]

    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    # plt.gca().set_facecolor('lightblue')  # Use any color name, hex code, or RGB value

    ax.set_extent([-125, -66.5, 24, 49.5], crs=ccrs.PlateCarree())  # US extent

    # Add map features
    # ax.add_feature(cfeature.COASTLINE)
    # ax.add_feature(cfeature.BORDERS, linestyle=':')
        # ax.add_feature(cfeature.STATES, linestyle=':')
        # import code; code.interact(local=dict(globals(), **locals()))
        # Plot all points at once
    sc = ax.scatter(
        longitudes, latitudes,  # Longitude and latitude arrays
        c=feats_ds,  # RGB colors
        s=1,                    # Marker size
        transform=ccrs.PlateCarree(),
        edgecolor='none'        # Speed up rendering by removing edges
    )

    plt.tight_layout()  # Avoid using this

    # plt.title(f"{args.location_model_name}", fontsize=20,pad=0)
    plt.show()
    # save output
    op_path = op_file_name
    print('Saving image to: ' + op_path)
    plt.savefig(op_path, bbox_inches='tight', dpi=300, pad_inches=0)
    # plt.imsave(sc, op_path)
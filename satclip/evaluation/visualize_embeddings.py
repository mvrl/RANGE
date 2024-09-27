"""
Extracts features from a trained network for each geo location, performs 
dimensionality reduction, and generates an output image.

Different seed values will result in different mappings of locations to colors. 
"""

import torch
import numpy as np
# import datasets
import matplotlib.pyplot as plt
import os
from sklearn import decomposition
from skimage import exposure
import json
# import utils
# import models
import argparse
#local import 
from .eval import LocationEncoder

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
    parser.add_argument('--location_model_name', type=str, default='RANF_HILO', help='Name of the location model')
    parser.add_argument('--ranf_db', type=str, default='/projects/bdec/adhakal2/hyper_satclip/data/models/ranf/ranf_satmae_db.npz', help='Path to the RANF database')
    return parser.parse_args()
# params - specify model of interest here

if __name__ == '__main__':
    args = get_args()
    seed = 2001
    # with open('paths.json', 'r') as f:
    #     paths = json.load(f)
    num_ds_dims = 3
    # model_path = 'pretrained_models/model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt' 

    op_file_name = f'/projects/bdec/adhakal2/hyper_satclip/data/eval_data/eval_results/{args.location_model_name}'+ '_ica.png'

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    # train_params = torch.load(eval_params['model_path'], map_location='cpu')
    model = LocationEncoder(args)
    model.eval()
    # if train_params['params']['input_enc'] in ['env', 'sin_cos_env']:
    #     raster = datasets.load_env()
    # else:
    #     raster = None
    # enc = utils.CoordEncoder(train_params['params']['input_enc'], raster=raster)

    # load ocean mask
    mask = np.load('/projects/bdec/adhakal2/hyper_satclip/data/eval_data/ocean_mask.npy')
    mask_inds = np.where(mask.reshape(-1) == 1)[0]

    locs = coord_grid(mask.shape)
    locs = locs[mask_inds, :]
    #this is in the form lon, lat
    locs = torch.from_numpy(locs).double()
    # import code; code.interact(local=dict(globals(), **locals()))
    with torch.no_grad():
        feats = model(locs,scale=None).cpu().numpy()

    # standardize the features
    f_mu = feats.mean(0)
    f_std = feats.std(0)
    feats = feats - f_mu
    feats = feats / f_std
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

    # convert into image
    op_im = np.ones((mask.shape[0]*mask.shape[1], num_ds_dims))
    op_im[mask_inds] = feats_ds
    op_im = op_im.reshape((mask.shape[0], mask.shape[1], num_ds_dims))

    # save output
    op_path = op_file_name
    print('Saving image to: ' + op_path)
    plt.imsave(op_path, (op_im*255).astype(np.uint8))
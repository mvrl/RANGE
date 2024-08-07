import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from einops import repeat
import argparse
#local imports
from ..fit import SAPCLIP_PCME, SAPCLIP
from ..load_satclip import get_satclip

## this script is used to evaluate the performance of the model on the ecoregion and biome dataset
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--scale', type=str, default='pool')
    parser.add_argument('--do_satclip', action='store_true')
    parser.add_argument('--ckpt_path', type=str, default='/scratch/a.dhakal/hyper_satclip/logs/SAPCLIP/pcme_models/epoch=813-val_loss=4609.ckpt')
    parser.add_argument('--region_type', type=str, default='BIOME_NAME') 
    parser.add_argument('--pcme_sapclip', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    device = args.device
    scale = args.scale
    do_satclip = True

    df_train = pd.read_csv('/scratch/a.dhakal/hyper_satclip/data/eval_data/ecoregion_train.csv')
    df_test = pd.read_csv('/scratch/a.dhakal/hyper_satclip/data/eval_data/ecoregion_val.csv')
    ckpt_path = args.ckpt_path

    df = pd.concat([df_train, df_test])

    labels = pd.factorize(df[args.region_type])[0]
    ytrain = labels[:len(df_train)]
    ytest = labels[len(df_train):]

    coords = df[['X', 'Y']].values
    coords = torch.from_numpy(coords).double().cuda()

    #load ckpt and load pretrained model
    if args.pcme_sapclip:
        model = SAPCLIP_PCME.load_from_checkpoint(ckpt_path).to(device)
    else:
        model = SAPCLIP.load_from_checkpoint(ckpt_path).to(device)
    sapclip_encoder = model.model.eval().cuda()
    satclip_encoder = get_satclip().location

    for param in sapclip_encoder.parameters():
        param.requires_grad = False

    for param in satclip_encoder.parameters():
        param.requires_grad = False

    # prepare input data
    loc = coords
    # import code; code.interact(local=dict(globals(), **locals()))

    map_scale = {1:torch.tensor([1,0,0]), 3:torch.tensor([0,1,0]), 5:torch.tensor([0,0,1])}
    
    if scale == 'pool':
        scale_1 = map_scale[1]
        scale_1 = repeat(scale_1, 'd -> b d', b=len(loc)).cuda()
        scale_3 = map_scale[3]
        scale_3 = repeat(scale_3, 'd -> b d', b=len(loc)).cuda()
        scale_5 = map_scale[5]
        scale_5 = repeat(scale_5, 'd -> b d', b=len(loc)).cuda() 
        # generate sapclip embeddings
        #instead of computing sapclip embeddings for individual scale, compute for all scales and average the embeddings
        loc_embeddings = sapclip_encoder.encode_location(coords=loc, scale=scale_1)
        loc_mu_1 = loc_embeddings[0].detach().cpu().numpy()
        loc_sigma_1 = torch.exp(loc_embeddings[1]).detach().cpu().numpy()
        loc_embeddings = sapclip_encoder.encode_location(coords=loc, scale=scale_3)
        loc_mu_3 = loc_embeddings[0].detach().cpu().numpy()
        loc_sigma_3 = torch.exp(loc_embeddings[1]).detach().cpu().numpy()
        loc_embeddings = sapclip_encoder.encode_location(coords=loc, scale=scale_5)
        loc_mu_5 = loc_embeddings[0].detach().cpu().numpy()
        loc_sigma_5 = torch.exp(loc_embeddings[1]).detach().cpu().numpy()
        import code; code.interact(local=dict(globals(), **locals()))

        loc_mu = (loc_mu_1 + loc_mu_3 + loc_mu_5)/3
        xtrain_sapclip = loc_mu[:len(df_train)]
        xtest_sapclip = loc_mu[len(df_train):]
        scaler = MinMaxScaler()
        xtrain_sapclip = scaler.fit_transform(xtrain_sapclip)
        xtest_sapclip = scaler.transform(xtest_sapclip)
        clf_sapclip = RidgeClassifierCV(alphas=(0.1, 1.0, 10.0), cv=10)
        clf_sapclip.fit(xtrain_sapclip, ytrain)
        print(f'The pooled sapclip score is: {clf_sapclip.score(xtest_sapclip, ytest)}')
    else:
        scale = int(scale)
        scale = map_scale[scale]
        scale = repeat(scale, 'd -> b d', b=len(loc)).cuda()
        loc_embeddings = sapclip_encoder.encode_location(coords=loc, scale=scale)
        loc_mu = loc_embeddings[0].detach().cpu().numpy()
        loc_sigma = torch.exp(loc_embeddings[1]).detach().cpu().numpy()
        xtrain_sapclip = loc_mu[:len(df_train)]
        xtest_sapclip = loc_mu[len(df_train):]
        scaler = MinMaxScaler()
        xtrain_sapclip = scaler.fit_transform(xtrain_sapclip)
        xtest_sapclip = scaler.transform(xtest_sapclip)
        clf_sapclip = RidgeClassifierCV(alphas=(0.1, 1.0, 10.0), cv=10)
        clf_sapclip.fit(xtrain_sapclip, ytrain)
        print(f'The sapclip score for scale {args.scale} is: {clf_sapclip.score(xtest_sapclip, ytest)}')


    #generate satclip embeddings
    if args.do_satclip:
        loc_embeddings = satclip_encoder(loc).detach().cpu().numpy()
        xtrain = loc_embeddings[:len(df_train)]
        xtest = loc_embeddings[len(df_train):]
        scaler = MinMaxScaler()
        xtrain = scaler.fit_transform(xtrain)
        xtest = scaler.transform(xtest)

        clf = RidgeClassifierCV(alphas=(0.1, 1.0, 10.0), cv=10)
        #clf = RandomForestClassifier(n_estimators=100)

        clf.fit(xtrain, ytrain)
        print(f'The satclip score is: {clf.score(xtest, ytest)}')
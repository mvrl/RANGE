import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from einops import repeat

#local imports
from ..fit import SAPCLIP_PCME
from ..load_satclip import get_satclip


device = 'cuda'

df_train = pd.read_csv('/scratch/a.dhakal/hyper_satclip/data/eval_data/ecoregion_train.csv')
df_test = pd.read_csv('/scratch/a.dhakal/hyper_satclip/data/eval_data/ecoregion_val.csv')

df = pd.concat([df_train, df_test])

labels = pd.factorize(df['BIOME_NAME'])[0]
ytrain = labels[:len(df_train)]
ytest = labels[len(df_train):]

coords = df[['X', 'Y']].values
coords = torch.from_numpy(coords).double().cuda()

#model = LocationBind(train_dataset=None, val_dataset=None)
ckpt_path = '/scratch/a.dhakal/hyper_satclip/logs/SAPCLIP/pcme_models/epoch=813-val_loss=4609.ckpt'

#load ckpt and load pretrained model
model = SAPCLIP_PCME.load_from_checkpoint(ckpt_path).to(device)
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
loc_mu = (loc_mu_1 + loc_mu_3 + loc_mu_5)/2
xtrain_sapclip = loc_mu[:len(df_train)]
xtest_sapclip = loc_mu[len(df_train):]
scaler = MinMaxScaler()
xtrain_sapclip = scaler.fit_transform(xtrain_sapclip)
xtest_sapclip = scaler.transform(xtest_sapclip)
clf_sapclip = RidgeClassifierCV(alphas=(0.1, 1.0, 10.0), cv=10)
clf_sapclip.fit(xtrain_sapclip, ytrain)
print(f'The sapclip score is: {clf_sapclip.score(xtest_sapclip, ytest)}')


#generate satclip embeddings
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
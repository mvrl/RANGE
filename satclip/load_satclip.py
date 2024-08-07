import code
import sys 
# sys.path.append('./satclip')
import torch


from .main_old import SatCLIPLightningModule
import torch.nn as nn
import lightning as L
import numpy as np
#local imports
from .loss_old import SatCLIPLoss
from .model_old import SatCLIP

def get_satclip(ckpt_path='/scratch/a.dhakal/hyper_satclip/satclip/location_model/satclip-vit16-l40.ckpt', device='cuda', return_all=False):
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt['hyper_parameters'].pop('eval_downstream')
    ckpt['hyper_parameters'].pop('air_temp_data_path')
    ckpt['hyper_parameters'].pop('election_data_path')
    lightning_model = SatCLIPLightningModule(**ckpt['hyper_parameters']).to(device)
    lightning_model.load_state_dict(ckpt['state_dict'])
    lightning_model.eval()
    geo_model = lightning_model.model
    return geo_model

if __name__ == '__main__':
    model = get_satclip()
    device = 'cuda'

    c = torch.randn(32, 2) # Represents a batch of 32 locations (lon/lat)

    model = get_satclip() #Only loads location encoder by default
    location_model = model.location
    location_model.eval()
    # import code; code.interact(local=dict(globals(), **locals()))
    with torch.no_grad():
        emb  = location_model(c.double().to(device)).detach().cpu()
    print(emb.shape)
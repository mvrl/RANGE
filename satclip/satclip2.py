import code
import sys 
# sys.path.append('./satclip')
import torch


from .main_old import SatCLIPLightningModule
import torch.nn as nn
import lightning as L
import numpy as np
#local imports
from .loss import SatCLIPLoss
from .model import SatCLIP

def get_satclip(ckpt_path, device='cuda', return_all=False):
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt['hyper_parameters'].pop('eval_downstream')
    ckpt['hyper_parameters'].pop('air_temp_data_path')
    ckpt['hyper_parameters'].pop('election_data_path')
    lightning_model = SatCLIPLightningModule(**ckpt['hyper_parameters']).to(device)
    lightning_model.load_state_dict(ckpt['state_dict'])
    lightning_model.eval()
    geo_model = lightning_model.model
    vision_model = 
    return geo_model
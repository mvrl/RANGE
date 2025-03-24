import numpy as np
import pandas as pd
import torch


from .main.utils import *
from .main.models import *

def get_csp(path='/projects/bdec/adhakal2/hyper_satclip/satclip/location_models/csp/model_dir/model_fmow/model_fmow_gridcell_0.0010_32_0.1000000_1_512_gelu_UNSUPER-contsoftmax_0.000050_1.000_1_0.100_TMP1.0000_1.0000_1.0000.pth.tar'):
    pretrained_csp = torch.load(path)

    params = pretrained_csp['params']
    loc_enc = get_model(
                            train_locs = None,
                            params = params,
                            spa_enc_type = params['spa_enc_type'],
                            num_inputs = params['num_loc_feats'],
                            num_classes = params['num_classes'],
                            num_filts = params['num_filts'],
                            num_users = params['num_users'],
                            device = params['device'])

    model = LocationImageEncoder(loc_enc = loc_enc,
                        train_loss = params["train_loss"],
                        unsuper_loss = params["unsuper_loss"],
                        cnn_feat_dim = params["cnn_feat_dim"],
                        spa_enc_type = params["spa_enc_type"]).to(params['device'])

    model.load_state_dict(pretrained_csp['state_dict'])

    return model

if __name__ == '__main__':
    path = '/projects/bdec/adhakal2/hyper_satclip/satclip/location_models/csp/model_dir/model_fmow/model_fmow_gridcell_0.0010_32_0.1000000_1_512_gelu_UNSUPER-contsoftmax_0.000050_1.000_1_0.100_TMP1.0000_1.0000_1.0000.pth.tar'
    model = get_csp(path)
    import code; code.interact(local=dict(globals(), **locals()))
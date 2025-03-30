# this script provide an easy way to load some of the commonly used location encoders including RANGE:
# RANGE, RANGE+
# SatCLIP
# GeoCLIP
# CSP, CSP_Inat
# SINR
# TaxaBind
# Direct
# Cartesian
# Wrap
import torch
from argparse import Namespace
from .range import LocationEncoder

def load_model(model_name='RANGE+', pretrained_dir='./pretrained', device='cuda', **kwargs):
    """
    Load the specified location encoder model.

    Args:
        model_name (str): The name of the model to load. Options are 'RANGE', 'RANGE+', 'SatCLIP', 'GeoCLIP', 'CSP', 'CSP_INat', 'SINR', 'TaxaBind', 'Direct', 'Cartesian_3D', and 'Wrap'.
        model_path (str): The path to the model file. Please look in the range.py file to see the expected directory structure.
        **kwargs:
            - Additional arguments for the model:
                    db_path: required for RANGE and RANGE+ pointing to the RANGE database path.
                    beta: required for RANGE+ model. defaults to 0.5.

    Returns:
        LocationEncoder: The loaded location encoder model.
    """
    if 'RANGE' in model_name:
        assert 'db_path' in kwargs, "db_path is required for RANGE model."
        db_path = kwargs.get('db_path')
        
        if 'beta' in kwargs:
            beta = kwargs.get('beta')
        else:
            beta = 0.5
    else:
        db_path = None
        beta = None
    #create namespace object
    args = Namespace(location_model_name=model_name, pretrained_dir=pretrained_dir,device=device,
                      range_db=db_path, beta=beta)
    #load the model
    model = LocationEncoder(args)
    model.eval()
    model.to(device)
    return model

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pretrained_dir = './pretrained'
    model_name = 'RANGE+'
    beta = 0.5
    db_path = '/projects/bdec/adhakal2/range/data/models/ranf/ranf_satmae_db.npz'
    rangep_model = load_model(model_name=model_name, model_path=pretrained_dir,
                               device=device, db_path=db_path, beta=beta)
    a = torch.rand(10000,2).double().to(device)
    import code; code.interact(local=dict(globals(), **locals()))
    

    


    
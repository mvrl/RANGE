from .gps2vec import *
import torch
import numpy as np
     
def get_gps2vec(locations,basedir,model='visual'):
    nrows = 20
    ncols = 20
    sigma = 20000
    if model=='visual':
      modeldir=basedir+"/models_visual"
      flag = 0
    elif model=='tag':
      modeldir=basedir+"/models_tag"
      flag = 1
    else:
      raise ValueError('Invalid model')
    out = []
    for location in locations:
      if location[0] <= -80 or location[0] >= 84:
         continue
      geofea = georep(location,modeldir,nrows,ncols,sigma,flag)
      out.append(np.asarray(geofea))
    return np.asarray(out, dtype=object)

if __name__ == '__main__':
    import code; code.interact(local=dict(globals(), **locals()))
    c = torch.Tensor([[-74.0060, 40.7128], [-118.2437, 34.0522]])  # Represents a batch of 2 locations (lon/lat)
    basedir = '/projects/bdec/adhakal2/hyper_satclip/satclip/location_models/GPS2Vec'
    model = 'visual'
    emb = get_gps2vec(np.flip(c.numpy(),1),basedir,model=model)
    print(emb)

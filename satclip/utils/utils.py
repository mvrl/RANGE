import numpy as np
import pymap3d as pm
import torch



def bounding_box_from_circle(lat_center, lon_center, radius = 1000,
    disable_latitude_compensation=False):
  '''
  radius is in meters determined at the equator

  warning: doesn't handle the poles or the 180th meridian very well, it might loop give a bad bounding box
   should probably define a check to make sure the radius isn't too big
  '''

  thetas = np.linspace(0,2*np.pi, 5)
  x, y = radius*np.cos(thetas), radius*np.sin(thetas)


  if not disable_latitude_compensation:
    # use tangent plane boxes, defined in meters at location
    lat, lon, alt = pm.enu2geodetic(x, y, 0, lat_center, lon_center, 0)
  else:
    # use lat-lon boxes, defined in meters at equator
    lat, lon, alt = pm.enu2geodetic(x, y, 0, 0, 0, 0)
    lat = lat + lat_center
    lon = lon + lon_center

  b,t = lat[3], lat[1]
  l,r = lon[2], lon[0]

  return l,b,r,t

def sample_gaussian_tensors(mu, logsigma, num_samples):
    eps = torch.randn(mu.size(0), num_samples, mu.size(1), dtype=mu.dtype, device=mu.device)
    samples = eps.mul(torch.exp(logsigma.unsqueeze(1))).add_(mu.unsqueeze(1))
    return samples
    

if __name__ == '__main__':
    import code; code.interact(local=dict(globals(), **locals()))
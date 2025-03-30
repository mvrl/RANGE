import numpy as np
import torch
import math

def sample_gaussian_tensors(mu, logsigma, num_samples):
    eps = torch.randn(mu.size(0), num_samples, mu.size(1), dtype=mu.dtype, device=mu.device)
    samples = eps.mul(torch.exp(logsigma.unsqueeze(1))).add_(mu.unsqueeze(1))
    return samples
    
#change lat_lon in radians to cartesian coordinates
def rad_to_cart(locations):
    x = np.cos(locations[:,1]) * np.cos(locations[:,0])
    y = np.cos(locations[:,1]) * np.sin(locations[:,0])
    z = np.sin(locations[:,1])
    xyz = np.stack([x, y, z], axis=1)
    return xyz

def my_sigmoid(x):
    return 1/(1+np.exp(-x))
#inflection point defines at which distance we want to weight 0.5

def shifted_sigmoid(a, inflection_point=15):
    shifted = a-inflection_point
    return 1-my_sigmoid(shifted)

def compute_haversine(X, Y, radians=False):
    EARTH_RADIUS=6371
    lon_1 = X[:,0]
    lat_1 = X[:,1]
    lon_2 = Y[:,0]
    lat_2 = Y[:,1]
    if not radians:
        lon_1 = lon_1 * math.pi/180
        lat_1 = lat_1 * math.pi/180
        lon_2 = lon_2 * math.pi/180
        lat_2 = lat_2 * math.pi/180
    #compute the distance
    a = np.sin((lat_2-lat_1)/2)**2 + np.cos(lat_1)*np.cos(lat_2)*np.sin((lon_2-lon_1)/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = EARTH_RADIUS*c
    return d

# def bounding_box_from_circle(lat_center, lon_center, radius = 1000,
#     disable_latitude_compensation=False):
#   '''
#   radius is in meters determined at the equator

#   warning: doesn't handle the poles or the 180th meridian very well, it might loop give a bad bounding box
#    should probably define a check to make sure the radius isn't too big
#   '''

#   thetas = np.linspace(0,2*np.pi, 5)
#   x, y = radius*np.cos(thetas), radius*np.sin(thetas)


#   if not disable_latitude_compensation:
#     # use tangent plane boxes, defined in meters at location
#     lat, lon, alt = pm.enu2geodetic(x, y, 0, lat_center, lon_center, 0)
#   else:
#     # use lat-lon boxes, defined in meters at equator
#     lat, lon, alt = pm.enu2geodetic(x, y, 0, 0, 0, 0)
#     lat = lat + lat_center
#     lon = lon + lon_center

#   b,t = lat[3], lat[1]
#   l,r = lon[2], lon[0]

#   return l,b,r,t

if __name__ == '__main__':
    import code; code.interact(local=dict(globals(), **locals()))
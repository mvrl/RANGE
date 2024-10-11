import numpy as np
import os
import argparse
from cuml.cluster import AgglomerativeClustering, KMeans

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from collections import Counter

import torch

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--data_path', type=str, default='/projects/bdec/adhakal2/hyper_satclip/data/models/ranf/ranf_satmae_db.npz')
    args.add_argument('--num_clusters', type=int, default=200)
    return args.parse_args()

if __name__ == '__main__':
    args = get_args()
    data = np.load(args.data_path)
    # Unpack the columns
    locations = torch.tensor(data['locs'])             # List of location data
    image_embeddings = torch.tensor(data['image_embeddings'])  # Stack image_embeddings into a 2D array
    satclip_embeddings = torch.tensor(data['satclip_embeddings'])  # Stack satclip_embeddings into a 2D array

    # Compute pairwise distances between image embeddings
    satclip_embeddings = satclip_embeddings / satclip_embeddings.norm(p=2, dim=1, keepdim=True)
    #perform clusterning
    clustering =  KMeans(n_clusters=args.num_clusters, max_iter=10000)
    clusters = clustering.fit_predict(satclip_embeddings.numpy())
    #aggregate the data by clusters
    unique_clusters = np.unique(clusters)
    clustered_locations = []
    clustered_satclip_means = []
    clustered_image_means = []
    cluster_sizes = []

    for cluster in unique_clusters:
        # Get indices of samples in this cluster
        cluster_indices = np.where(clusters == cluster)[0]
        
        # Aggregate locations
        cluster_locations = locations[cluster_indices].tolist()
        clustered_locations.append(cluster_locations)
        
        # Compute mean of satclip_embeddings for the cluster
        mean_satclip = satclip_embeddings[cluster_indices].mean(dim=0)
        clustered_satclip_means.append(mean_satclip)
        
        # Compute mean of image_embeddings for the cluster
        mean_image = image_embeddings[cluster_indices].mean(dim=0)
        clustered_image_means.append(mean_image)
        
        # Store the cluster size
        cluster_size = len(cluster_indices)
        cluster_sizes.append(cluster_size)
    
    #concatenate the data
    
    clustered_satclip_means = torch.stack(clustered_satclip_means)
    clustered_image_means = torch.stack(clustered_image_means)
    cluster_sizes = np.array(cluster_sizes).reshape(-1,1)
    #pad the locations
    max_cluster_size = np.max(cluster_sizes)
    padding_element = [None,None]
    for i in range(len(clustered_locations)):
        padding_needed = max_cluster_size - len(clustered_locations[i])
        #append [0,0] to the outer list
        if padding_needed > 0:
            clustered_locations[i].extend([padding_element]*padding_needed)
    clustered_locations = np.array(clustered_locations)
    #save the data
    np.savez(f'{os.path.dirname(args.data_path)}/{args.num_clusters}-clustered_data.npz',
             clustered_locations=clustered_locations,
             clustered_satclip_means=clustered_satclip_means.numpy(),
             clustered_image_means=clustered_image_means.numpy(),
             cluster_sizes=cluster_sizes)
    import code; code.interact(local=dict(globals(), **locals()))


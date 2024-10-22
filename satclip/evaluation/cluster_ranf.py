import numpy as np
import os
import argparse
from cuml.cluster import AgglomerativeClustering, KMeans, DBSCAN

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from collections import Counter

import torch

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--data_path', type=str, default='/projects/bdec/adhakal2/hyper_satclip/data/models/ranf/ranf_satmae_db.npz')
    args.add_argument('--num_clusters', type=int, default=200)
    args.add_argument('--ranf_model', type=str, default='GeoCLIP')
    args.add_argument('--cluster_type', choices=['kmeans', 'hierarchical', 'dbscan'])
    return args.parse_args()




if __name__ == '__main__':
    args = get_args()
    data = np.load(args.data_path)
    # Unpack the columns
    locations = torch.tensor(data['locs'])             # List of location data
    image_embeddings = torch.tensor(data['image_embeddings'])  # Stack image_embeddings into a 2D array
    
    if args.ranf_model == 'GeoCLIP':
        lowres_embeddings = torch.tensor(data['geoclip_embeddings'])  # Stack satclip_embeddings into a 2D array
    elif args.ranf_model == 'SatCLIP':
        lowres_embeddings = torch.tensor(data['satclip_embeddings'])
    else:    
        raise ValueError('Invalid model type. Choose between GeoCLIP and SatCLIP')
    # Compute pairwise distances between image embeddings
    lowres_embeddings = lowres_embeddings / lowres_embeddings.norm(p=2, dim=1, keepdim=True)
    #perform clusterning
    if args.cluster_type == 'kmeans':
        clustering =  KMeans(n_clusters=args.num_clusters, max_iter=10000, verbose=4)
        clusters = clustering.fit_predict(lowres_embeddings.numpy())
    elif args.cluster_type == 'dbscan':
        clustering = DBSCAN(eps=1.02, min_samples=512)
        clusters = clustering.fit_predict(lowres_embeddings.numpy(), out_dtype=np.int64)
    elif args.cluster_type == 'hierarchical':
        clustering = AgglomerativeClustering(n_clusters=args.num_clusters, metric='cosine', connectivity='pairwise')    
    else:
        raise ValueError('Invalid clustering type')

    
    import code; code.interact(local=dict(globals(), **locals()))
    #aggregate the data by clusters
    #get the unique clusters
    unique_clusters = np.unique(clusters)
    #get the count per cluster
    count_cluster = Counter(clusters)
    clustered_locations = []
    clustered_lowres_means = []
    clustered_image_means = []
    cluster_sizes = []

    for cluster in unique_clusters:
        # Get indices of samples in this cluster
        if count_cluster[cluster] < 100:
            continue
        cluster_indices = np.where(clusters == cluster)[0]
        
        # Aggregate locations
        cluster_locations = locations[cluster_indices].tolist()
        clustered_locations.append(cluster_locations)
        
        # Compute mean of satclip_embeddings for the cluster
        mean_lowres = lowres_embeddings[cluster_indices].mean(dim=0)
        clustered_lowres_means.append(mean_lowres)
        
        # Compute mean of image_embeddings for the cluster
        mean_image = image_embeddings[cluster_indices].mean(dim=0)
        clustered_image_means.append(mean_image)
        
        # Store the cluster size
        cluster_size = len(cluster_indices)
        cluster_sizes.append(cluster_size)
    
    #concatenate the data
    
    clustered_satclip_means = torch.stack(clustered_lowres_means)
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
    if args.ranf_model=='GeoCLIP':
        np.savez(f'{os.path.dirname(args.data_path)}/{args.cluster_type}-{args.num_clusters}-clustered_db.npz',
             locs=clustered_locations,
             geoclip_embeddings=clustered_satclip_means.numpy(),
             image_embeddings=clustered_image_means.numpy(),
             cluster_sizes=cluster_sizes)
    elif args.ranf_model=='SatCLIP':
        np.savez(f'{os.path.dirname(args.data_path)}/{args.cluster_type}-{args.num_clusters}-clustered_db.npz',
             locs=clustered_locations,
             satclip_embeddings=clustered_satclip_means.numpy(),
             image_embeddings=clustered_image_means.numpy(),
             cluster_sizes=cluster_sizes)

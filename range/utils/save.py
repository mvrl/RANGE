import numpy as np
import os
import torch
from tqdm import tqdm


def save_embeddings(args, train_loader, val_loader, location_model):
    embeddings_dir = os.path.join(args.embeddings_dir, args.location_model_name)
    #check if directory already exist for this model
    if not os.path.exists(embeddings_dir):
        print(f'Creating new directory {embeddings_dir}')
        os.makedirs(embeddings_dir)
    #create train and val path
    train_path = os.path.join(embeddings_dir, f'{args.task_name}_train.npz')
    val_path = os.path.join(embeddings_dir, f'{args.task_name}_val.npz')
    #freeze the model
    location_model.eval()
    location_model = location_model.to(args.device)
    with torch.no_grad():
        #first get the embeddings for the train data
        coords_list = []
        embeddings_list = []
        y_list = []
        for i, data in tqdm(enumerate(train_loader)):
            coords, y = data
            coords = coords.to(args.device)
            try:
                location_embeddings = location_model(coords).cpu().numpy()
            except AttributeError:
                location_embeddings = location_model(coords)
            coords = coords.cpu().numpy()
            y = y.cpu().numpy()
            coords_list.append(coords)
            embeddings_list.append(location_embeddings)
            y_list.append(y)
        #save the embeddings
        np.savez(train_path, coords=np.concatenate(coords_list, axis=0), embeddings=np.concatenate(embeddings_list, axis=0), y=np.concatenate(y_list, axis=0))
        print(f'File saved to {train_path}')
        #reset the lists
        coords_list = []
        embeddings_list = []
        y_list = []
        #compute embeddings for validation data
        for i, data in tqdm(enumerate(val_loader)):
            coords, y = data
            coords = coords.to(args.device)
            try:
                location_embeddings = location_model(coords).cpu().numpy()
            except AttributeError:
                location_embeddings = location_model(coords)
            coords = coords.cpu().numpy()
            y = y.cpu().numpy()
            coords_list.append(coords)
            embeddings_list.append(location_embeddings)
            y_list.append(y)
        #save the embeddings
        np.savez(val_path, coords=np.concatenate(coords_list, axis=0), embeddings=np.concatenate(embeddings_list, axis=0), y=np.concatenate(y_list, axis=0))
        print(f'File saved to {train_path} and {val_path}')
# This module is used to save the embeddings from the location models into a numpy file: using --eval_type=save_embeddings 
# This module is also used to evaluate the embeddings for the downstream tasks: using --eval_type=evaluate_npz
# The location encoder that can be evaluated using this module are:RANGE, RANGE+, SatCLIP, GeoCLIP, TaxaBind, CSP, CSP_INat, SINR, Direct, Cartesian_3D, sphere2vec, theory
# the desired location encoder can be selected using the --location_model_name argument

import numpy as np
import math
import argparse
import os
import sys

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

#loading different location models

from .location_models.satclip.load import get_satclip
from geoclip import LocationEncoder as GeoCLIP #input as lat,long
from rshf.sinr import SINR
from rshf.sinr import preprocess_locs as preprocess_sinr
from .location_models.csp.load_csp import get_csp
from .location_models.GPS2Vec.get_gps2vec import get_gps2vec
from .location_models.sphere2vec.sphere2vec import get_sphere2vec
from .location_models.satclip.positional_encoding.theory import Theory
from .location_models.satclip.positional_encoding.wrap import Wrap


#local import from utils 
from .utils.save import save_embeddings
from .utils.evaluate import evaluate_npz
from .utils.load_dataset import get_dataset
from .utils.utils import rad_to_cart


EARTH_RADIUS=6371
def get_args():
    parser = argparse.ArgumentParser(description='code for evaluating the embeddings')
    #location model arguments
    parser.add_argument('--location_model_name', type=str, help='Name of the location model', default='SatCLIP')
    parser.add_argument('--range_db', type=str, default='/projects/bdec/adhakal2/range/data/models/ranf/ranf_satmae_db.npz')
    parser.add_argument('--range_model', type=str, default='', choices=['SatCLIP',''])
    parser.add_argument('--pretrained_dir', type=str, default='/projects/bdec/adhakal2/hyper_satclip/satclip/location_models')
    parser.add_argument('--beta', type=float, default=0.5, help='Beta value for RANGE_COMBINED')  
    parser.add_argument('--task_name', type=str, help='Name of the task', default='biome')
    parser.add_argument('--eval_dir', type=str, help='Path to the evaluation data directory', default='/projects/bdec/adhakal2/hyper_satclip/data/eval_data')
    parser.add_argument('--batch_size', type=int, help='Batch size', default=5000)
    parser.add_argument('--num_workers', type=int, help='Number of workers', default=6)

    #logging arguments
    parser.add_argument('--log_dir', type=str, help='Path to the log directory', default='./')
    parser.add_argument('--run_name', type=str, help='Name of the run', default='downstream_eval')
    parser.add_argument('--project_name', type=str, help='Name of the project', default='Donwstream Evaluation')
    parser.add_argument('--wandb_mode', type=str, help='Mode of wandb', default='online')

    #saving embeddings
    parser.add_argument('--embeddings_dir', type=str, default='/projects/bdec/adhakal2/range/data/saved_embeddings')
    #eval type
    parser.add_argument('--eval_type', type=str, default='evaluate_npz', choices=['save_embeddings', 'evaluate_npz'])
    #device arguments
    parser.add_argument('--accelerator', type=str, default='gpu',choices=['gpu','cpu'], help='Accelerator to use')
    args = parser.parse_args()

    return args


#Dummy location encoder
class DummyLocationEncoder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, coords):
        return coords

class LocationEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.location_model_name = args.location_model_name
        #get the appropriate model
        #SatCLIP for encoding location
        if self.location_model_name == 'SatCLIP':
            print('Using SatCLIP')
            ckpt = os.path.join(args.pretrained_dir,'satclip/satclip-vit16-l40.ckpt')
            self.loc_model = get_satclip(
                    ckpt, device=args.device).double()
            self.location_feature_dim = 256  
        #satclip for encoding location and image
        elif self.location_model_name == 'GeoCLIP':
            print('Using GeoCLIP')
            self.loc_model = GeoCLIP().double()
            self.location_feature_dim = 512
        #taxabind for encoding location
        elif self.location_model_name == 'TaxaBind':
            print('Using TaxaBind')
            self.loc_model = GeoCLIP().double()
            ckpt = torch.load(args.pretrained_dir, 'taxabind/patched_location_encoder.pt', map_location=args.device)
            self.loc_model.load_state_dict(ckpt)
            self.location_feature_dim = 512
        #CSP_FMOW
        elif self.location_model_name == 'CSP':
            print('Using CSP-FMOW')
            self.loc_model = get_csp(path=os.path.join(args.pretrained_dir,'csp/fmow/model_fmow_gridcell_0.0010_32_0.1000000_1_512_gelu_UNSUPER-contsoftmax_0.000050_1.000_1_0.100_TMP1.0000_1.0000_1.0000.pth.tar'))
            self.location_feature_dim = 256
        #CSP INAT
        elif self.location_model_name == 'CSP_INat':
            print('Using CSP-IN75.97lkjhat')
            self.loc_model = get_csp(path=os.path.join(args.pretrained_dir,'csp/inat/model_inat_2018_gridcell_0.0010_32_0.1000000_1_512_leakyrelu_UNSUPER-contsoftmax_0.000500_1.000_1_1.000_TMP20.0000_1.0000_1.0000.pth.tar'))
            self.location_feature_dim = 256
        #Direct
        elif self.location_model_name == 'Direct':
            print('Using Direct Encoding')
            self.loc_model = DummyLocationEncoder()
            self.location_feature_dim = 2
        #Cartesian_3D
        elif self.location_model_name == 'Cartesian_3D':
            print('Using Cartesian_3D')
            self.loc_model = DummyLocationEncoder()
            self.location_feature_dim = 3
        #Theory
        elif self.location_model_name == 'Theory':
            print('Using Theory')
            self.loc_model = Theory(frequency_num=32, min_radius=1)
            self.location_feature_dim = 0
        #Wrap
        elif self.location_model_name == 'Wrap':
            print('Using Wrap')
            self.loc_model = Wrap()
            self.location_feature_dim = 4
        #sphere2vec
        elif 's2vec' in self.location_model_name:
            print('Using sphere2vec')
            self.s2vec_type = self.location_model_name.split('_')[-1]
            self.loc_model = get_sphere2vec(name=self.s2vec_type)
            if self.s2vec_type == 'spherem':
                self.location_feature_dim = 256
            elif self.s2vec_type == 'spherec':
                self.location_feature_dim = 288
            elif self.s2vec_type == 'spheremplus':
                self.location_feature_dim = 512
            elif self.s2vec_type == 'spherecplus':
                self.location_feature_dim = 192
        #SINR
        elif self.location_model_name == 'SINR':
            print('Using SINR')
            self.loc_model = SINR().double()
            self.location_feature_dim = 256
        elif 'RANGE' in self.location_model_name:
            #load the database
            range_db = np.load(args.range_db, allow_pickle=True)
            self.db_locs_latlon = range_db['locs'].astype(np.float32)
        
            #get satcilp location encoder
            ckpt = os.path.join(args.pretrained_dir,'range/satclip-vit16-l40.ckpt')
            self.loc_model = get_satclip(
                    ckpt, device=args.device).double()
            self.db_satclip_embeddings = range_db['satclip_embeddings'].astype(np.float32)
            self.location_feature_dim = 1024 + 256
            
            #normalize the embeddings    
            self.db_satclip_embeddings = torch.tensor(self.db_satclip_embeddings/np.linalg.norm(self.db_satclip_embeddings, ord=2, axis=1, keepdims=True))
            self.db_high_resolution_satclip_embeddings = torch.tensor(range_db['image_embeddings'].astype(np.float32))
            
            #convert lon, lat to radians
            self.db_locs = self.db_locs_latlon * math.pi/180
            #convert to cartesian coordinates
            self.db_locs_xyz = rad_to_cart(self.db_locs)

            #send to cuda            
            if args.device=='cuda':
                self.db_satclip_embeddings = torch.tensor(self.db_satclip_embeddings).to(args.device)
                self.db_locs_xyz = torch.tensor(self.db_locs_xyz).to(args.device)
            #use RANGE model
            if self.location_model_name == 'RANGE':    
                temp = 15.0
                self.args.temp = temp
                print(f'Using RANGE with temperature {self.args.temp}')
            # use RANGE+ model
            elif self.location_model_name == 'RANGE+':
                semantic_temp = 12.0
                geo_temp = 40.0
                self.args.geo_temp = geo_temp
                self.args.temp = semantic_temp
                print(f'Using RANGE+ with temperatures {self.args.temp} and {self.args.geo_temp}')
            else:
                raise ValueError('Unimplemented RANGE model')
            
            
        else:
            raise NotImplementedError(f'{self.location_model_name} not implemented')
        self.loc_model.eval()
        for params in self.loc_model.parameters():
            params.requires_grad=False

    #return the location embeddings
    def forward(self, coords):
        ###############  RANGE  #####################
        if 'RANGE' in self.location_model_name:
            #get the satclip embeddings for the given location
            curr_loc_embeddings = self.loc_model(coords).to(self.args.device)
            #normalize the embeddings and compute similarity
            curr_loc_embeddings = curr_loc_embeddings/curr_loc_embeddings.norm(p=2, dim=-1, keepdim=True)
            high_res_similarity = curr_loc_embeddings.float() @ self.db_satclip_embeddings.t()
            #scale similarity using temp and convert to probabilities
            high_res_similarity = torch.nn.functional.softmax(high_res_similarity * self.args.temp, dim=-1) #batch, num_db
            #compute the highres embeddings as a weighted sum of databased embeddings
            high_res_embeddings = high_res_similarity @ self.db_high_resolution_satclip_embeddings.to(self.args.device) #batch, 1024

            #concatenate rich image features with low res location features
            if self.location_model_name=='RANGE':
                # RANGE embeddings
                loc_embeddings = np.concatenate((high_res_embeddings.cpu(), curr_loc_embeddings.cpu()), axis=1)
            elif self.location_model_name=='RANGE+':
                # RANGE+ embeddings
                query_locations_latlon = coords.cpu().numpy()
                #convert to radians
                query_locations = query_locations_latlon * math.pi/180
                #convert to cartesian coordinates
                query_locations_xyz = torch.tensor(rad_to_cart(query_locations))
                #get the similarity between the query locations and the database locations
                angular_similarity = query_locations_xyz.float().to(self.args.device) @ self.db_locs_xyz.T
                #remove data that is further than certain point
                #scale the similarity and convert to probablilities
                angular_similarity = nn.functional.softmax(angular_similarity * self.args.geo_temp, dim=-1)
                #get the scale averaged high res embeddings
                angular_high_res_embeddings = angular_similarity @ self.db_high_resolution_satclip_embeddings.to(self.args.device)
                #compute the average between the two high res embeddings
                averaged_high_res_embeddings = (1-self.args.beta)*angular_high_res_embeddings + self.args.beta*high_res_embeddings
                #concatenate this with the low res values
                loc_embeddings = np.concatenate((averaged_high_res_embeddings.cpu(), curr_loc_embeddings.cpu()), axis=1)
            else:
                raise ValueError('Unimplemented RANGE model')
        ###############  SatCLIP #####################
        elif self.location_model_name == 'SatCLIP':
            loc_embeddings = self.loc_model(coords)
        ###############  GeoCLIP #####################
        elif self.location_model_name == 'GeoCLIP':
            coords = coords[:,[1,0]]
            loc_embeddings = self.loc_model(coords)
        #################  CSP #####################
        elif 'CSP' in self.location_model_name:
            loc_embeddings = self.loc_model(coords, return_feats=True)
        #################  SINR #####################
        elif self.location_model_name == 'SINR':
            coords = preprocess_sinr(coords)
            loc_embeddings = self.loc_model(coords)
        ################ taxaBind #####################
        elif self.location_model_name == 'TaxaBind':
            coords = coords[:,[1,0]]
            loc_embeddings = self.loc_model(coords)
        ################ training-free methods #####################
        elif self.location_model_name == 'Direct':
            coords_rad = coords * math.pi/180
            loc_embeddings = self.loc_model(coords_rad)
        elif self.location_model_name == 'Cartesian_3D':
            coords_rad = coords * math.pi/180
            coords_cart = rad_to_cart(coords_rad.cpu().numpy())
            loc_embeddings = self.loc_model(coords_cart)
        elif self.location_model_name == 'Theory':
            loc_embeddings = self.loc_model(coords)
        elif self.location_model_name == 'Wrap':
            loc_embeddings = self.loc_model(coords)
        
        elif 's2vec' in self.location_model_name:
            loc_embeddings = self.loc_model(coords)
        else:
            raise NotImplementedError(f'{self.location_model_name} not implemented')
        return loc_embeddings


if __name__ == '__main__':
    args = get_args()
    #set device
    if args.accelerator == 'gpu':
        args.device = 'cuda'
    elif args.accelerator == 'cpu':
        args.device = 'cpu'
    
    #initialize the pretrained location model
    location_model = LocationEncoder(args)
    #get the output dimension of the location model
    location_feature_dim = location_model.location_feature_dim
    #initialize the dataset
    train_loader, val_loader, num_classes = get_dataset(args)
    #precompute embeddings and save them
    if args.eval_type == 'save_embeddings':
        print('Saving npz files for embeddings...')
        save_embeddings(args,train_loader, val_loader,location_model)
        train_path = os.path.join(args.eval_dir, 'train_embeddings.npz')
    
    #run the eval for precomputed embeddings
    elif args.eval_type == 'evaluate_npz':
        print('Evaluating embeddings from precomputed npz files')
        acc = evaluate_npz(args)
        acc = np.round(acc, 3)
        print(f'Accuracy: {acc}')
        sys.stderr.write(f'Accuracy: {acc}')

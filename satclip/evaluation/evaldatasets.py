from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import pandas as pd
import os
import numpy as np
import json

##local import
from ..datamodules.s2geo_dataset import S2GeoDataModule 
#srikumar
class Biome_Dataset(Dataset):
    def __init__(self, data_path, scale=0):
        map_scale = {0:torch.tensor([]),1:torch.tensor([1,0,0]),3:torch.tensor([0,1,0]),5:torch.tensor([0,0,1])}
        self.curr_scale = map_scale[scale].double()
        print(f'Using scale {scale} for Dataset')
        self.data_path = data_path
        self.train_data_path = os.path.join(data_path,'ecoregion_train.csv')
        self.val_data_path = os.path.join(data_path,'ecoregion_val.csv')
        self.train_df = pd.read_csv(self.train_data_path)
        self.val_df = pd.read_csv(self.val_data_path) 
        #join the two dataframes
        self.df = pd.concat([self.train_df,self.val_df])
        self.df.dropna(subset=['BIOME_NAME'],inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        self.label, self.label_map = pd.factorize(self.df['BIOME_NAME'])
        self.loc = self.df[['X', 'Y']].values
        self.num_classes = self.df['BIOME_NAME'].nunique() 

    def __getitem__(self, index):
        loc =  torch.from_numpy(self.loc[index]).double()
        label = self.label[index]
        return loc,self.curr_scale,label        

    def __len__(self):
        return len(self.label)

#srikumar
class Eco_Dataset(Dataset):
    def __init__(self, data_path, scale=0):
        map_scale = {0:torch.tensor([]),1:torch.tensor([1,0,0]),3:torch.tensor([0,1,0]),5:torch.tensor([0,0,1])}
        self.curr_scale = map_scale[scale].double()
        print(f'Using scale {scale} for Dataset')
        self.data_path = data_path
        self.train_data_path = os.path.join(data_path,'ecoregion_train.csv')
        self.val_data_path = os.path.join(data_path,'ecoregion_val.csv')
        self.train_df = pd.read_csv(self.train_data_path)
        self.val_df = pd.read_csv(self.val_data_path) 
        #join the two dataframes
        self.df = pd.concat([self.train_df,self.val_df])
        self.df.dropna(subset=['ECO_NAME'],inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        self.label, self.label_map = pd.factorize(self.df['ECO_NAME'])
        self.loc = self.df[['X', 'Y']].values
        self.num_classes = self.df['ECO_NAME'].nunique() 

    def __getitem__(self, index):
        loc =  torch.from_numpy(self.loc[index]).double()
        label = self.label[index]
        return loc,self.curr_scale,label        

    def __len__(self):
        return len(self.label)


class Temp_Dataset(Dataset):
    def __init__(self, data_path, scale=0):
        map_scale = {0:torch.tensor([]),1:torch.tensor([1,0,0]),3:torch.tensor([0,1,0]),5:torch.tensor([0,0,1])}
        self.curr_scale = map_scale[scale].double()
        print(f'Using scale {scale} for Dataset')
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.df.dropna(subset=['meanT'],inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        self.loc = self.df[['Lon', 'Lat']].values
        self.label = self.df['meanT']
        self.num_classes = 0

    def __getitem__(self, index):
        loc =  torch.from_numpy(self.loc[index]).double()
        label = torch.tensor(self.label[index]).double()
        return loc,self.curr_scale,label        

    def __len__(self):
        return len(self.label)

#https://www.kaggle.com/datasets/camnugent/california-housing-prices
class Housing_Dataset(Dataset):
    def __init__(self, data_path, scale=0):
        map_scale = {0:torch.tensor([]),1:torch.tensor([1,0,0]),3:torch.tensor([0,1,0]),5:torch.tensor([0,0,1])}
        self.curr_scale = map_scale[scale].double()
        print(f'Using scale {scale} for Dataset')
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.df.dropna(subset=['median_house_value'],inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        self.loc = self.df[['longitude', 'latitude']].values
        self.label = self.df['median_house_value']
        self.num_classes = 0

    def __getitem__(self, index):
        loc =  torch.from_numpy(self.loc[index]).double()
        label = torch.tensor(self.label[index]).double()
        return loc,self.curr_scale,label        

    def __len__(self):
        return len(self.label)

#https://codeocean.com/capsule/6456296/tree/v2
class Elevation_Dataset(Dataset):
    def __init__(self, data_path, scale=0):
        map_scale = {0:torch.tensor([]),1:torch.tensor([1,0,0]),3:torch.tensor([0,1,0]),5:torch.tensor([0,0,1])}
        self.curr_scale = map_scale[scale].double()
        print(f'Using scale {scale} for Dataset')
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.df.dropna(subset=['elevation'],inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        self.loc = self.df[['lon', 'lat']].values
        self.label = self.df['elevation']
        self.num_classes = 0
        
    def __getitem__(self, index):
        loc =  torch.from_numpy(self.loc[index]).double()
        label = torch.tensor(self.label[index]).double()
        return loc,self.curr_scale,label        

    def __len__(self):
        return len(self.label)

#https://codeocean.com/capsule/6456296/tree/v2
class Population_Dataset(Dataset):
    def __init__(self, data_path, scale=0):
        map_scale = {0:torch.tensor([]),1:torch.tensor([1,0,0]),3:torch.tensor([0,1,0]),5:torch.tensor([0,0,1])}
        self.curr_scale = map_scale[scale].double()
        print(f'Using scale {scale} for Dataset')
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.df.dropna(subset=['population'],inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        self.loc = self.df[['lon', 'lat']].values
        self.label = self.df['population']
        self.num_classes = 0

    def __getitem__(self, index):
        loc =  torch.from_numpy(self.loc[index]).double()
        label = torch.tensor(self.label[index]).double()
        return loc,self.curr_scale,np.log(1+label)        

    def __len__(self):
        return len(self.label)

class NaBirdDataset(Dataset):
    def __init__(self, data_path,scale=0,type='val'):
        map_scale = {0:torch.tensor([]),1:torch.tensor([1,0,0]),3:torch.tensor([0,1,0]),5:torch.tensor([0,0,1])}
        self.curr_scale = map_scale[scale].double()
        print(f'Using scale {scale} for Dataset')
        self.data_path = data_path
        
        #load train data
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        if type=='train':
            data = data['train']
        elif type=='val':
            data = data['test']

        self.df = pd.DataFrame(data)
        self.df.dropna(subset=['class_id'],inplace=True)

        self.df.reset_index(drop=False, inplace=True)
        #get metadata
        meta_table = pd.DataFrame(list(self.df['ebird_meta']))
        self.df['lon'] = meta_table['lon']
        self.df['lat'] = meta_table['lat']
        #drop nan values in lon and lat
        self.df.dropna(subset=['lon','lat'],inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        #assign lon, lat to loc
        self.loc = self.df[['lon', 'lat']].values
        self.label = self.df['class_id']

        self.num_classes = len(self.label.unique())
    
    def __getitem__(self, index):
        loc =  torch.from_numpy(self.loc[index]).double()
        label = self.label[index]
        return loc,self.curr_scale,label

    def __len__(self):
        return len(self.label)    

class INatMiniDataset(Dataset):
    def __init__(self, data_path,scale=0,type='val'):
        map_scale = {0:torch.tensor([]),1:torch.tensor([1,0,0]),3:torch.tensor([0,1,0]),5:torch.tensor([0,0,1])}
        self.curr_scale = map_scale[scale].double()
        print(f'Using scale {scale} for Dataset')
        if type=='train':
            self.data_path = os.path.join(data_path,'train_mini.json')
        elif type=='val':
            self.data_path = os.path.join(data_path,'val.json')
        
        #load train data
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        self.sample_df = pd.DataFrame(data['images'])
        self.annot_df = pd.DataFrame(data['annotations'])
        self.df =self.sample_df.merge(self.annot_df, on='id')
        #drop all nan rows and reset index 
        self.df.dropna(subset=['category_id', 'latitude', 'longitude'],inplace=True)
        self.df.reset_index(drop=False, inplace=True)
        
        #assign lon, lat to loc
        self.loc = self.df[['longitude', 'latitude']].values
        self.label = self.df['category_id']

        self.num_classes = len(self.label.unique())
    
    def __getitem__(self, index):
        loc =  torch.from_numpy(self.loc[index]).double()
        label = self.label[index]
        return loc,self.curr_scale,label

    def __len__(self):
        return len(self.label)     
        

if __name__ == '__main__':

    inat_mini_path = '/projects/bdec/adhakal2/hyper_satclip/data/eval_data/inat_mini'
    inat_mini_dataset = INatMini(inat_mini_path, scale=0,type='train')
    # nabird_data_path = '/projects/bdec/adhakal2/hyper_satclip/data/eval_data/inat/geo_prior_data/data/nabirds/nabirds_with_loc_2019.json'
    # nabird_dataset = NaBirdDataset(nabird_data_path, scale=0)
    # import code; code.interact(local=dict(globals(), **locals()))
    # biome_data_path = '/projects/bdec/adhakal2/hyper_satclip/data/eval_data'
    # biome_dataset = Biome_Dataset(biome_data_path, scale=1)
    # temp_data_path = '/projects/bdec/adhakal2/hyper_satclip/data/eval_data/temp.csv'
    # temp_dataset = Temp_Dataset(temp_data_path,scale=3)

    # housing_data_path = '/projects/bdec/adhakal2/hyper_satclip/data/eval_data/housing.csv'
    # housing_dataset = Housing_Dataset(housing_data_path, scale=5)

    # elevation_data_path = '/projects/bdec/adhakal2/hyper_satclip/data/eval_data/elevation.csv'
    # elevation_dataset = Elevation_Dataset(elevation_data_path, scale=0)
    
    # population_data_path = '/projects/bdec/adhakal2/hyper_satclip/data/eval_data/population.csv'
    # population_dataset = Population_Dataset(population_data_path, scale=0)
    

    
    # ##### test #########
    # import pandas as pd
    # lats = []
    # lons = []
    # labels = []
    # for i, data in enumerate(dataset_val):
    #     loc, scale, label = data
    #     lat = loc[1].cpu().numpy()
    #     lon = loc[0].cpu().numpy()
    #     lats.append(lat)
    #     lons.append(lon)
    #     labels.append(label)

    # train_dict = {'X':lons, 'Y':lats, 'ECO_NAME':labels}
    # df = pd.DataFrame(train_dict)
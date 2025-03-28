from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import pandas as pd
import os
import numpy as np
import json

##local import
from .checkerboarddataset import CheckerDataset
#srikumar
class Biome_Dataset(Dataset):
    def __init__(self, data_path):
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
        return loc, label        

    def __len__(self):
        return len(self.label)

#srikumar
class Eco_Dataset(Dataset):
    def __init__(self, data_path):
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
        return loc,label        

    def __len__(self):
        return len(self.label)

class Inat_Dataset(Dataset):
    def __init__(self, data_path, type='train'):
        self.data_path = data_path
        self.train_data_path = os.path.join(data_path,'inat2018_train.csv')
        
        self.val_data_path = os.path.join(data_path,'inat2018_val.csv')
        self.train_df = pd.read_csv(self.train_data_path)
        self.train_df.drop('Unnamed: 0', inplace=True, axis=1)
        self.val_df = pd.read_csv(self.val_data_path) 
        self.val_df = self.val_df[['lon', 'lat', 'class']]
        train_len = len(self.train_df)
        val_len = len(self.val_df)
        #join the two dataframes
        self.df = pd.concat([self.train_df,self.val_df], ignore_index=True)
        # self.df.dropna(subset=['lon','lat', 'class'],inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        # self.label, self.label_map = pd.factorize(self.df['class'])
        self.label = self.df['class'].values
        self.loc = self.df[['lon', 'lat']].values
        if type=='train':
            self.label = self.label[:train_len]
            self.loc = self.loc[:train_len]
        elif type == 'val':
            self.label = self.label[train_len:]
            self.loc = self.loc[train_len:]
        
        self.num_classes = self.df['class'].nunique() 

    def __getitem__(self, index):
        loc =  torch.from_numpy(self.loc[index]).double()
        label = self.label[index]
        return loc,label        

    def __len__(self):
        return len(self.label)

class Country_Dataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.df.dropna(subset=['country', 'lat','lon'],inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        self.loc = self.df[['lon', 'lat']].values
        self.label = self.df['country']
        self.num_classes = self.df['country'].nunique()
    
    def __getitem__(self, index):
        loc =  torch.from_numpy(self.loc[index]).double()
        label = self.label[index]
        return loc,label

    def __len__(self):
        return len(self.label)

class Ocean_Dataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.df.dropna(subset=['land', 'lat','lon'],inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        self.loc = self.df[['lon', 'lat']].values
        self.label = self.df['land']
        self.num_classes = self.df['land'].nunique()
    
    def __getitem__(self, index):
        loc =  torch.from_numpy(self.loc[index]).double()
        label = self.label[index]
        return loc,label

    def __len__(self):
        return len(self.label)

class CSVDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.loc = self.df[['lon', 'lat']].values
        self.label = self.df.index
        self.num_classes = 0

    def __getitem__(self, index):
        loc =  torch.from_numpy(self.loc[index]).double()
        label = self.label[index]
        return loc,label
    
    def __len__(self):
        return len(self.label)


class Temp_Dataset(Dataset):
    def __init__(self, data_path):
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
        return loc,label        

    def __len__(self):
        return len(self.label)

#https://www.kaggle.com/datasets/camnugent/california-housing-prices
class Housing_Dataset(Dataset):
    def __init__(self, data_path):
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
        return loc,label        

    def __len__(self):
        return len(self.label)

#https://codeocean.com/capsule/6456296/tree/v2
class Elevation_Dataset(Dataset):
    def __init__(self, data_path):
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
        return loc,label        

    def __len__(self):
        return len(self.label)

#dataset for ERA5 data
class ERA5_Dataset(Dataset):
    def __init__(self, data_path, group='air_temp_m'):
        self.group=group
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.df.dropna(subset=[group],inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        self.loc = self.df[['Longitude', 'Latitude']].values
        self.label = self.df[group]
        self.num_classes = 0

    def __getitem__(self, index):
        loc =  torch.from_numpy(self.loc[index]).double()
        label = torch.tensor(self.label[index]).double()
        return loc,label
    
    def __len__(self):
        return len(self.label)


#https://codeocean.com/capsule/6456296/tree/v2
class Population_Dataset(Dataset):
    def __init__(self, scale=0):
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
        return loc,np.log(1+label)        

    def __len__(self):
        return len(self.label)
    
        
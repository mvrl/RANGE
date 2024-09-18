from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import pandas as pd

#srikumar
class Biome_Dataset(Dataset):
    def __init__(self, data_path, scale):
        map_scale = {0:torch.tensor([]),1:torch.tensor([1,0,0]),3:torch.tensor([0,1,0]),5:torch.tensor([0,0,1])}
        self.curr_scale = map_scale[scale].double()
        print(f'Using scale {scale} for Dataset')
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
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
    def __init__(self, data_path, scale):
        map_scale = {0:torch.tensor([]),1:torch.tensor([1,0,0]),3:torch.tensor([0,1,0]),5:torch.tensor([0,0,1])}
        self.curr_scale = map_scale[scale].double()
        print(f'Using scale {scale} for Dataset')
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.df.dropna(subset=['ECO_BIOME_'],inplace=True)
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
    def __init__(self, data_path, scale):
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
    def __init__(self, data_path, scale):
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
    def __init__(self, data_path, scale):
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
    def __init__(self, data_path, scale):
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
        return loc,self.curr_scale,label        

    def __len__(self):
        return len(self.label)
    

if __name__ == '__main__':
    biome_data_path = '/projects/bdec/adhakal2/hyper_satclip/data/eval_data/ecoregion_train.csv'
    biome_dataset = Biome_Dataset(biome_data_path, scale=1)
    temp_data_path = '/projects/bdec/adhakal2/hyper_satclip/data/eval_data/temp.csv'
    temp_dataset = Temp_Dataset(temp_data_path,scale=3)

    housing_data_path = '/projects/bdec/adhakal2/hyper_satclip/data/eval_data/housing.csv'
    housing_dataset = Housing_Dataset(housing_data_path, scale=5)

    elevation_data_path = '/projects/bdec/adhakal2/hyper_satclip/data/eval_data/elevation.csv'
    elevation_dataset = Elevation_Dataset(elevation_data_path, scale=0)
    
    population_data_path = '/projects/bdec/adhakal2/hyper_satclip/data/eval_data/population.csv'
    population_dataset = Population_Dataset(population_data_path, scale=0)
    import code; code.interact(local=dict(globals(), **locals()))

    
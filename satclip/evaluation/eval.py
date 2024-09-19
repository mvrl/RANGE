import lightning.pytorch as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger 
from lightning.pytorch.callbacks import ModelCheckpoint
import torchmetrics
from huggingface_hub import hf_hub_download
from torch.utils.data import random_split, DataLoader

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import RidgeClassifierCV, RidgeCV

import argparse
from einops import repeat
import os
from tqdm import tqdm 
import sys
#local import 
from ..utils.load_model import load_checkpoint
from .evaldatasets import Biome_Dataset, Eco_Dataset, Temp_Dataset, Housing_Dataset, Elevation_Dataset, Population_Dataset
#loading different location models
from ..load import get_satclip
from geoclip import LocationEncoder as GeoCLIP #input as lat,long

import warnings

# Suppress all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

def get_args():
    parser = argparse.ArgumentParser(description='code for evaluating the embeddings')
    #location model arguments
    parser.add_argument('--ckpt_path', type=str, help='Path to the pretrained model',
    default='/projects/bdec/adhakal2/hyper_satclip/logs/SAPCLIP/0tflzztx/checkpoints/epoch=162-acc_eco=0.000.ckpt')
    parser.add_argument('--location_model_name', type=str, help='Name of the location model', default='SAPCLIP')
    
    #dataset arguments
    parser.add_argument('--task_name', type=str, help='Name of the task', default='population',
                        choices=['biome', 'ecoregion', 'temperature', 'housing', 'elevation', 'population'])
    parser.add_argument('--eval_dir', type=str, help='Path to the evaluation data directory', default='/projects/bdec/adhakal2/hyper_satclip/data/eval_data')
    parser.add_argument('--scale', type=int, help='Scale for the location', choices=[0,1,3,5], default=0)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=64)
    parser.add_argument('--num_workers', type=int, help='Number of workers', default=6)
    
    #logging arguments
    parser.add_argument('--log_dir', type=str, help='Path to the log directory', default='/projects/bdec/adhakal2/hyper_satclip/logs/downstream')
    parser.add_argument('--run_name', type=str, help='Name of the run', default='downstream_eval')
    parser.add_argument('--project_name', type=str, help='Name of the project', default='Donwstream Evaluation')
    parser.add_argument('--wandb_mode', type=str, help='Mode of wandb', default='online')
    
    #downstream model argumetns
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--dev_run', action='store_true', help='Run the model in dev mode')
    parser.add_argument('--learning_rate', type=float, default=1e-03)
    #saving embeddings
    parser.add_argument('--embeddings_dir', type=str, default='/projects/bdec/adhakal2/hyper_satclip/data/eval_data/embeddings')
    #eval type
    parser.add_argument('--eval_type', type=str, default='evaluate_npz', choices=['save_embeddings', 'evaluate_raw', 'evaluate_npz'])
    args = parser.parse_args()

    return args

def save_embeddings(args, train_loader, val_loader, location_model):
    feature_dim = location_model.location_feature_dim
    embeddings_dir = os.path.join(args.embeddings_dir, args.location_model_name)
    #check if directory already exist for this model
    if not os.path.exists(embeddings_dir):
        print(f'Creating new directory {embeddings_dir}')
        os.makedirs(embeddings_dir)
    #create train and val path
    train_path = os.path.join(embeddings_dir, f'{args.task_name}_scale-{args.scale}_train.npz')
    val_path = os.path.join(embeddings_dir, f'{args.task_name}_scale-{args.scale}_val.npz')
    #freeze the model
    location_model.eval()
    with torch.no_grad():
        #first get the embeddings for the train data
        coords_list = []
        scale_list = []
        embeddings_list = []
        y_list = []
        for i, data in tqdm(enumerate(train_loader)):
            coords, scale, y = data
            coords = coords.to(args.device)
            scale = scale.to(args.device)
            location_embeddings = location_model(coords, scale).cpu().numpy()
            scale = scale.cpu().numpy()
            coords = coords.cpu().numpy()
            y = y.cpu().numpy()
            coords_list.append(coords)
            scale_list.append(scale)
            embeddings_list.append(location_embeddings)
            y_list.append(y)
        #save the embeddings
        np.savez(train_path, coords=np.concatenate(coords_list, axis=0), scale=np.concatenate(scale_list, axis=0), embeddings=np.concatenate(embeddings_list, axis=0), y=np.concatenate(y_list, axis=0))
        print(f'File saved to {train_path}')
        #reset the lists
        coords_list = []
        scale_list = []
        embeddings_list = []
        y_list = []
        #compute embeddings for validation data
        for i, data in tqdm(enumerate(val_loader)):
            coords, scale, y = data
            coords = coords.to(args.device)
            scale = scale.to(args.device)
            location_embeddings = location_model(coords, scale).cpu().numpy()
            scale = scale.cpu().numpy()
            coords = coords.cpu().numpy()
            y = y.cpu().numpy()
            coords_list.append(coords)
            scale_list.append(scale)
            embeddings_list.append(location_embeddings)
            y_list.append(y)
        #save the embeddings
        np.savez(val_path, coords=np.concatenate(coords_list, axis=0), scale=np.concatenate(scale_list, axis=0), embeddings=np.concatenate(embeddings_list, axis=0), y=np.concatenate(y_list, axis=0))
        print(f'File saved to {train_path} and {val_path}')

def evaluate_npz(args):
    train_path = os.path.join(args.embeddings_dir, args.location_model_name, args.task_name+'_scale-'+str(args.scale)+'_train.npz')
    val_path = os.path.join(args.embeddings_dir, args.location_model_name, args.task_name+'_scale-'+str(args.scale)+'_val.npz')
    assert os.path.exists(train_path), f'Train embeddings file does not exist: {train_path}'
    assert os.path.exists(val_path), f'Val embeddings file does not exist: {val_path}'
    #get training data
    train_data = np.load(train_path)
    train_embeddings = train_data['embeddings']
    train_labels = train_data['y']
    #get validation data
    val_data = np.load(val_path)
    val_embeddings = val_data['embeddings']
    val_labels = val_data['y']
    #decide the model
    if args.task_name == 'ecoregion' or args.task_name == 'biome':
        print('Classification Model')
        clf = RidgeClassifierCV(alphas=(0.1, 1.0, 10.0), cv=10)
    else:
        print('Regression Model')
        clf = RidgeCV(alphas=(0.1, 1.0, 10.0), cv=10)
    #normalize the embeddings
    scaler = MinMaxScaler()
    train_embeddings = scaler.fit_transform(train_embeddings)
    val_embeddings = scaler.transform(val_embeddings)
    #run the classifier
    clf.fit(val_embeddings, val_labels)
    val_accuracy = clf.score(val_embeddings, val_labels)
    print(f'The validation set accuracy is {val_accuracy}')
    return val_accuracy

def get_dataset(args):
    generator = torch.Generator().manual_seed(42)
    if args.task_name == 'biome':
        data_path = args.eval_dir
        dataset = Biome_Dataset(data_path, args.scale)
        dataset_train, dataset_val = random_split(dataset, [0.8, 0.2], generator=generator)
        num_classes = dataset_train.dataset.num_classes
    elif args.task_name == 'ecoregion':
        data_path = args.eval_dir
        dataset = Eco_Dataset(data_path, args.scale)
        dataset_train, dataset_val = random_split(dataset, [0.8, 0.2], generator=generator)
        num_classes = dataset_train.dataset.num_classes
    elif args.task_name == 'temperature':
        data_path = os.path.join(args.eval_dir, 'temp.csv')
        dataset = Temp_Dataset(data_path, args.scale)
        dataset_train, dataset_val = random_split(dataset, [0.8, 0.2], generator=generator)
        num_classes = dataset_train.dataset.num_classes
    elif args.task_name == 'housing':
        data_path = os.path.join(args.eval_dir, 'housing.csv')
        dataset = Housing_Dataset(data_path, args.scale)
        dataset_train, dataset_val = random_split(dataset, [0.8, 0.2], generator=generator)
        num_classes = dataset_train.dataset.num_classes
    elif args.task_name == 'elevation':
        data_path = os.path.join(args.eval_dir, 'elevation.csv')
        dataset = Elevation_Dataset(data_path, args.scale)
        dataset_train, dataset_val = random_split(dataset, [0.8, 0.2], generator=generator)
        num_classes = dataset_train.dataset.num_classes
    elif args.task_name == 'population':
        data_path = os.path.join(args.eval_dir, 'population.csv')
        dataset = Population_Dataset(data_path, args.scale)
        dataset_train, dataset_val = random_split(dataset, [0.8, 0.2], generator=generator)
        num_classes = dataset_train.dataset.num_classes
    else:
        raise ValueError('Task name not recognized')
    
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=False)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)
    return train_loader, val_loader, num_classes

class LocationEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.location_model_name = args.location_model_name
        #get the appropriate model
        if self.location_model_name == 'SAPCLIP':
            print('Using SAPCLIP')
            self.loc_model = load_checkpoint(args.ckpt_path, args.device).model.eval()
            self.location_feature_dim = 256
        elif self.location_model_name == 'SatCLIP':
            print('Using SatCLIP')
            self.loc_model = get_satclip(
                    hf_hub_download("microsoft/SatCLIP-ViT16-L40", "satclip-vit16-l40.ckpt", force_download=False),
                device=args.device).double()
            self.location_feature_dim = 256  
        elif self.location_model_name == 'GeoCLIP':
            print('Using GeoCLIP')
            self.loc_model = GeoCLIP().double()
            self.location_feature_dim = 512
        elif self.location_model_name == 'TaxaBind':
            print('Using TaxaBind')
            self.loc_model = GeoCLIP().double()
            ckpt = torch.load('/projects/bdec/adhakal2/hyper_satclip/data/models/patched_location_encoder.pt', map_location=args.device)
            self.loc_model.load_state_dict(ckpt)
            self.location_feature_dim = 512

        else:
            raise NotImplementedError(f'{self.location_model_name} not implemented')

    #return the location embeddings
    def forward(self, coords, scale):
        if self.location_model_name == 'SAPCLIP':
            loc_embeddings = self.loc_model.encode_location(coords, scale)[0]
        elif self.location_model_name == 'SatCLIP':
            loc_embeddings = self.loc_model(coords)
        elif self.location_model_name == 'GeoCLIP':
            coords = coords[:,[1,0]]
            loc_embeddings = self.loc_model(coords)
        elif self.location_model_name == 'TaxaBind':
            coords = coords[:,[1,0]]
            loc_embeddings = self.loc_model(coords)
        else:
            raise NotImplementedError(f'{self.location_model_name} not implemented')
        return loc_embeddings

# NN for linear probe of embeddings for regression tasks
class RegressionNet(L.LightningModule):
    def __init__(self,
     location_model,
     input_dims: int=256,
     **kwargs):
        super().__init__()
        self.location_encoder = location_model.eval()
        #freeze the model
        for param in self.location_encoder.parameters():
            param.requires_grad=False
        
        self.hidden = torch.nn.Linear(input_dims, 256).double()
        self.linear = torch.nn.Linear(256, 1).double()
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.acc = torchmetrics.R2Score()
        #save all values for acc calculation
        self.true_labels = []
        self.predicted_labels = []


    def forward(self, coords, scale):
        location_embeddings = self.location_encoder(coords, scale)
        inter = torch.nn.functional.leaky_relu(self.hidden(location_embeddings))
        out = self.linear(inter)
        return out

    def shared_step(self, batch):
        coords, scale, y = batch
        y_hat = torch.squeeze(self(coords, scale))
        loss = self.criterion(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        coords, scale, y = batch
        y_hat = torch.squeeze(self(coords, scale))
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        #save all values for acc calculation
        self.true_labels.append(y)
        self.predicted_labels.append(y_hat)
        return loss
    
    def on_validation_epoch_end(self):
        predicted_labels = torch.cat(self.predicted_labels)
        true_labels = torch.cat(self.true_labels)
        # acc = self.acc(predicted_labels, true_labels)
        acc = 1-((predicted_labels - true_labels).pow(2).sum() / ((true_labels - true_labels.mean()).pow(2).sum() + 1e-8))
        loss = self.criterion(predicted_labels, true_labels)
        self.log('val_acc', acc, prog_bar=True)
        self.log('mse_loss', loss, prog_bar=True)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.linear.parameters(), lr=1e-3)

# NN for linear probe of embeddings for classification tasks
class ClassificationNet(L.LightningModule):
    def __init__(self,
     location_model,
     input_dims: int=256,
     num_classes: int=10,
     **kwargs):
        super().__init__()
        self.location_encoder = location_model.eval()
        #freeze the model
        for param in self.location_encoder.parameters():
            param.requires_grad=False
        
        self.hidden = torch.nn.Linear(input_dims, 256).double()
        self.linear = torch.nn.Linear(256, num_classes).double()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        #save all values for acc calculation
        self.true_labels = []
        self.predicted_labels = []


    def forward(self, coords, scale):
        location_embeddings = self.location_encoder(coords, scale)
        inter = torch.nn.functional.leaky_relu(self.hidden(location_embeddings))
        out = self.linear(inter)
        return out

    def shared_step(self, batch):
        coords, scale, y = batch
        y_hat = self(coords, scale)
        loss = self.criterion(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        coords, scale, y = batch
        y_hat = self(coords, scale)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        #save all values for acc calculation
        self.true_labels.append(y)
        y_hat = torch.argmax(y_hat, dim=1)
        self.predicted_labels.append(y_hat)
        return loss
    
    def on_validation_epoch_end(self):
        predicted_labels = torch.cat(self.predicted_labels)
        true_labels = torch.cat(self.true_labels)
        acc = self.acc(predicted_labels, true_labels)
        self.log('val_acc', acc, prog_bar=True)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.linear.parameters(), lr=args.learning_rate)


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
        print(f'Accuracy: {acc}')
        sys.stderr.write(f'Accuracy: {acc}')

    #run the model on raw data
    elif args.eval_type == 'evaluate_raw':
        print('Evaluating embeddings from raw data')
        if num_classes == 0:
            donwstream_task = 'regression'
            model = RegressionNet(location_model, location_feature_dim)
        
            downstream_task = 'classification'
            model = ClassificationNet(location_model, location_feature_dim, num_classes)
        
        #initialize the logger
        wandb_logger = WandbLogger(save_dir=args.log_dir, project=args.project_name,
        name=args.run_name, mode=args.wandb_mode)
        #initialize the checkpoint callback
        checkpoint_callback = ModelCheckpoint(monitor='val_acc', filename='{epoch}_{val_acc:.3f}', save_top_k=3, mode='max', save_last=True)
        #initialize the trainer
        if args.dev_run:
            print('Running Dev Mode!')
            trainer = L.Trainer(fast_dev_run=15, precision='64', max_epochs=args.max_epochs, strategy='ddp_find_unused_parameters_false',
                num_sanity_val_steps=1, accelerator=args.accelerator, check_val_every_n_epoch=1)
        else:
            print('Running Full Training!')
            trainer = L.Trainer(precision='64', max_epochs=args.max_epochs, strategy='ddp_find_unused_parameters_false',
                num_sanity_val_steps=1, accelerator=args.accelerator, check_val_every_n_epoch=1,
                logger=wandb_logger, callbacks=[checkpoint_callback], log_every_n_steps=1)

        trainer.fit(model, train_loader, val_loader)
    

    

   
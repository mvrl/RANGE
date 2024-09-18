import lightning.pytorch as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger 
from lightning.pytorch.callbacks import ModelCheckpoint
import torchmetrics
from torch.utils.data import random_split, DataLoader
from sklearn.preprocessing import MinMaxScaler
import argparse
from einops import repeat
import os
#local import 
from ..utils.load_model import load_checkpoint
from .evaldatasets import Biome_Dataset, Eco_Dataset, Temp_Dataset, Housing_Dataset, Elevation_Dataset, Population_Dataset


def get_args():
    parser = argparse.ArgumentParser(description='code for evaluating the embeddings')
    #location model arguments
    parser.add_argument('--ckpt_path', type=str, help='Path to the pretrained model',
    default='/projects/bdec/adhakal2/hyper_satclip/logs/SAPCLIP/0tflzztx/checkpoints/epoch=162-acc_eco=0.000.ckpt')
    parser.add_argument('--location_model_name', type=str, help='Name of the location model', default='SAPCLIP')
    #dataset arguments
    parser.add_argument('--task_name', type=str, help='Name of the task', default='population')
    parser.add_argument('--eval_dir', type=str, help='Path to the evaluation data directory', default='/projects/bdec/adhakal2/hyper_satclip/data/eval_data')
    parser.add_argument('--batch_size', type=int, help='Batch size', default=64)
    parser.add_argument('--num_workers', type=int, help='Number of workers', default=6)
    #logging arguments
    parser.add_argument('--log_dir', type=str, help='Path to the log directory', default='/projects/bdec/adhakal2/hyper_satclip/logs/downstream')
    parser.add_argument('--run_name', type=str, help='Name of the run', default='downstream_eval')
    parser.add_argument('--project_name', type=str, help='Name of the project', default='Donwstream Evaluation')
    parser.add_argument('--wandb_mode', type=str, help='Mode of wandb', default='online')
    #downstream model argumetns
    parser.add_argument('--device', type=str, help='Device to run the model', default='cuda')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--accelerator', type=str, default='gpu')
    # parser.add_argument('--devices', default=1)
    args = parser.parse_args()

    return args

#get the approp
def get_dataset(args):
    generator = torch.Generator().manual_seed(42)
    if args.task_name == 'biome':
        data_path_train = os.path.join(args.eval_dir, 'ecoregion_train.csv')
        data_path_val = os.path.join(args.eval_dir, 'ecoregion_val.csv')
        dataset_train = Biome_Dataset(data_path_train)
        dataset_val = Biome_Dataset(data_path_val)
        num_classes = dataset_train.num_classes
    elif args.task_name == 'ecoregion':
        data_path_train = os.path.join(args.eval_dir, 'ecoregion_train.csv')
        data_path_val = os.path.join(args.eval_dir, 'ecoregion_val.csv')
        dataset_train = Eco_Dataset(data_path_train)
        dataset_val = Eco_Dataset(data_path_val)
        num_classes = dataset_train.num_classes
    elif args.task_name == 'temperature':
        data_path = os.path.join(args.eval_dir, 'temp.csv')
        dataset = Temp_Dataset(data_path)
        dataset_train, dataset_val = random_split(dataset, [0.8, 0.2], generator=generator)
        num_classes = dataset_train.dataset.num_classes
    elif args.task_name == 'housing':
        data_path = os.path.join(args.eval_dir, 'housing.csv')
        dataset = Housing_Dataset(data_path)
        dataset_train, dataset_val = random_split(dataset, [0.8, 0.2], generator=generator)
        num_classes = dataset_train.dataset.num_classes
    elif args.task_name == 'elevation':
        data_path = os.path.join(args.eval_dir, 'elevation.csv')
        dataset = Elevation_Dataset(data_path)
        dataset_train, dataset_val = random_split(dataset, [0.8, 0.2], generator=generator)
        num_classes = dataset_train.dataset.num_classes
    elif args.task_name == 'population':
        data_path = os.path.join(args.eval_dir, 'population.csv')
        dataset = Population_Dataset(data_path)
        dataset_train, dataset_val = random_split(dataset, [0.8, 0.2], generator=generator)
        num_classes = dataset_train.dataset.num_classes
    else:
        raise ValueError('Task name not recognized')
    
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    return train_loader, val_loader, num_classes

class LocationEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.location_model_name = args.location_model_name
        #get the appropriate model
        if self.location_model_name == 'SAPCLIP':
            self.loc_model = load_checkpoint(args.ckpt_path, args.device).model.eval()
        else:
            raise NotImplementedError(f'{self.location_model_name} not implemented')

    #return the location embeddings
    def forward(self, coords, scale):
        if self.location_model_name == 'SAPCLIP':
            loc_embeddings = self.loc_model.encode_location(coords, scale)[0]
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
        
        self.linear = torch.nn.Linear(input_dims, 1).double()
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.acc = torchmetrics.R2Score()
        #save all values for acc calculation
        self.true_labels = []
        self.predicted_labels = []


    def forward(self, coords, scale):
        location_embeddings = self.location_encoder(coords, scale)
        out = self.linear(location_embeddings)
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
    
    def val_loss(self, batch, batch_idx):
        coords, scale, y = batch
        y_hat = self(coords, scale)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        #save all values for acc calculation
        self.true_labels.append(y)
        self.predicted_labels.append(y_hat)
        return loss
    
    def on_validation_epoch_end(self):
        predicted_labels = torch.cat(self.predicted_labels)
        true_labels = torch.cat(self.true_labels)
        acc = self.acc(predicted_labels, true_labels)
        self.log('val_acc', acc, prog_bar=True)
    
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
        
        self.linear = torch.nn.Linear(input_dims, num_classes).double()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        #save all values for acc calculation
        self.true_labels = []
        self.predicted_labels = []


    def forward(self, coords, scale):
        location_embeddings = self.location_encoder(coords, scale)
        out = self.linear(location_embeddings)
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
    
    def val_loss(self, batch, batch_idx):
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
        return torch.optim.Adam(self.linear.parameters(), lr=1e-3)


if __name__ == '__main__':
    args = get_args()
    #initialize the pretrained location model
    location_model = LocationEncoder(args)
    #initialize the dataset
    train_loader, val_loader, num_classes = get_dataset(args)
    #find the number of classes and type
    if num_classes == 0:
        donwstream_task = 'regression'
        model = RegressionNet(location_model, 256)
    else:
        downstream_task = 'classification'
        model = ClassificationNet(location_model, 256, num_classes)
    
    #initialize the logger
    wandb_logger = WandbLogger(save_dir=args.log_dir, project=args.project_name,
     name=f'{args.location_model_name}_{args.task_name}',mode=args.wandb_mode)
    #initialize the checkpoint callback
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', filename='{epoch}_{val_acc}:.3f', save_top_k=3, mode='max', save_last=True)
    #initialize the trainer
    trainer = L.Trainer(precision='64', max_epochs=args.max_epochs, strategy='ddp_find_unused_parameters_false',
    num_sanity_val_steps=1, accelerator=args.accelerator, check_val_every_n_epoch=1,
    logger=wandb_logger, callbacks=[checkpoint_callback])

    trainer.fit(model, train_loader, val_loader)
    
    #randomly generate some data
    coords = torch.rand(10, 2).double()
    scale = torch.tensor([0,0,1]).double()
    scale = repeat(scale, 'd -> b d', b=10).double()
    import code; code.interact(local=dict(globals(), **locals()))
    out = model(coords, scale)
    

   
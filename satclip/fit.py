import argparse
import os
from datetime import datetime
from argparse import ArgumentParser

import lightning.pytorch as L
import torch
import torch.nn.functional as F
# from datamodules.s2geo_dataset import S2GeoDataModule
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.cli import LightningCLI




#local imports 
from .model import SatCLIP_2
from .datamodules.sapclip_dataset import SAPCLIP_Dataset, get_split_dataset, SAPCLIP_Dataset_H5

torch.set_float32_matmul_precision('high')

def contrastive_loss(similarity, labels):
    #labels is class probablities of shape [N,C], where C is the number of classes 
    return F.cross_entropy(similarity, labels)

class SAPCLIP(L.LightningModule):
    def __init__(
        self,
        embed_dim=512,
        image_resolution=256,
        vision_layers='CLIP',
        vision_width=768,
        vision_patch_size=32,
        in_channels=4,
        le_type="grid",
        pe_type="siren",
        frequency_num=16,
        max_radius=260,
        min_radius=1,
        legendre_polys=16,
        harmonics_calculation="analytic",
        sh_embedding_dims=32,
        learning_rate=1e-4,
        weight_decay=0.01,
        num_hidden_layers=2,
        capacity=256,        
        loss_type='probablistic',
       anneal_T=0.001
    ) -> None:
        super().__init__()

        self.model = SatCLIP_2(
            embed_dim=embed_dim,
            image_resolution=image_resolution,
            vision_layers=vision_layers,
            vision_width=vision_width,
            vision_patch_size=vision_patch_size,
            in_channels=in_channels,
            le_type=le_type,
            pe_type=pe_type,
            frequency_num=frequency_num,
            max_radius=max_radius,
            min_radius=min_radius,
            legendre_polys=legendre_polys,
            harmonics_calculation=harmonics_calculation,
            sh_embedding_dims=sh_embedding_dims,
            num_hidden_layers=num_hidden_layers,
            capacity=capacity,
            device='cuda',
            loss_type=loss_type,
        )
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.anneal_T = anneal_T
        self.delta_beta = 2/self.anneal_T
        self.kld_wt=0
        self.save_hyperparameters()

    def anneal_beta(self):
        if self.global_step%self.anneal_T == 0:
            self.kld_wt = 0
        elif self.global_step%self.anneal_T>self.anneal_T/2:
            self.kld_wt = 1
        else:
            self.kld_wt+=self.delta_beta
    
    #increase upto 0.5 and drop back to 0
    def anneal_beta_2(self):
        if self.kld_wt>=0.35:
            self.kld_wt=0     
        else:
            self.kld_wt+=self.delta_beta

    #configure optimizers 
    def configure_optimizers(self):
        exclude = (
            lambda n, p: p.ndim < 2
            or "bn" in n
            or "ln" in n
            or "bias" in n
            or "logit_scale" in n
        )
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(self.model.named_parameters())
        gain_or_bias_params = [
            p for n, p in named_parameters if exclude(n, p) and p.requires_grad
        ]
        rest_params = [
            p for n, p in named_parameters if include(n, p) and p.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.0},
                {
                    "params": rest_params,
                    "weight_decay": self.weight_decay,
                },  # specify in configs/default.yaml
            ],
            lr=self.learning_rate,  # specify in configs/default.yaml
        )
        return optimizer
    
    #define the forward pass
    def forward(self, batch, batch_idx):
        # images = batch['image']
        # points = batch['point']
        # scale = batch['scale']        
        contrastive_loss, kld_loss = self.model(batch)

        return contrastive_loss, kld_loss

    def training_step(self, batch, batch_idx):
        contrastive_loss, kld_loss = self(batch, batch_idx)
        loss = contrastive_loss + self.kld_wt * kld_loss
        self.anneal_beta_2()
        self.log('train_contrastive_loss', contrastive_loss, batch_size=len(batch), prog_bar=True, sync_dist=True)
        self.log('train_kld_loss', kld_loss, batch_size=len(batch), prog_bar=True, sync_dist=True)
        self.log('train_loss', loss, batch_size=len(batch), prog_bar=True, sync_dist=True)
        self.log('beta', self.kld_wt, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        contrastive_loss, kld_loss = self(batch, batch_idx)
        loss = contrastive_loss + kld_loss
        self.log('val_contrastive_loss', contrastive_loss, batch_size=len(batch), prog_bar=True, sync_dist=True)
        self.log('val_kld_loss', kld_loss, batch_size=len(batch), prog_bar=True, sync_dist=True)
        self.log('val_loss', loss, batch_size=len(batch), prog_bar=True, sync_dist=True)
        return loss    

class SAPCLIP_PCME(L.LightningModule):
    def __init__(
        self,
        embed_dim=512,
        image_resolution=256,
        vision_layers='CLIP',
        vision_width=768,
        vision_patch_size=32,
        in_channels=4,
        le_type="grid",
        pe_type="siren",
        frequency_num=16,
        max_radius=260,
        min_radius=1,
        legendre_polys=16,
        harmonics_calculation="analytic",
        sh_embedding_dims=32,
        learning_rate=1e-4,
        weight_decay=0.01,
        num_hidden_layers=2,
        capacity=256,        
        loss_type='pcme',
        anneal_T=1000
    ) -> None:
        super().__init__()

        self.model = SatCLIP_2(
            embed_dim=embed_dim,
            image_resolution=image_resolution,
            vision_layers=vision_layers,
            vision_width=vision_width,
            vision_patch_size=vision_patch_size,
            in_channels=in_channels,
            le_type=le_type,
            pe_type=pe_type,
            frequency_num=frequency_num,
            max_radius=max_radius,
            min_radius=min_radius,
            legendre_polys=legendre_polys,
            harmonics_calculation=harmonics_calculation,
            sh_embedding_dims=sh_embedding_dims,
            num_hidden_layers=num_hidden_layers,
            capacity=capacity,
            device='cuda',
            loss_type=loss_type,
        )
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.anneal_T = anneal_T
        self.delta_beta = 2/self.anneal_T
        self.kld_wt=0
        self.save_hyperparameters()

    def anneal_beta(self):
        if self.global_step%self.anneal_T == 0:
            self.kld_wt = 0
        elif self.global_step%self.anneal_T>self.anneal_T/2:
            self.kld_wt = 1
        else:
            self.kld_wt+=self.delta_beta
    
    #increase upto 0.5 and drop back to 0
    def anneal_beta_2(self):
        if self.kld_wt>=0.35:
            self.kld_wt=0     
        else:
            self.kld_wt+=self.delta_beta

    #configure optimizers 
    def configure_optimizers(self):
        exclude = (
            lambda n, p: p.ndim < 2
            or "bn" in n
            or "ln" in n
            or "bias" in n
            or "logit_scale" in n
        )
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(self.model.named_parameters())
        gain_or_bias_params = [
            p for n, p in named_parameters if exclude(n, p) and p.requires_grad
        ]
        rest_params = [
            p for n, p in named_parameters if include(n, p) and p.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.0},
                {
                    "params": rest_params,
                    "weight_decay": self.weight_decay,
                },  # specify in configs/default.yaml
            ],
            lr=self.learning_rate,  # specify in configs/default.yaml
        )
        return optimizer
    
    #define the forward pass
    def forward(self, batch, batch_idx):      
        pcme_loss, pcme_dict = self.model(batch)

        return pcme_loss, pcme_dict

    def training_step(self, batch, batch_idx):
        pcme_loss, pcme_dict = self(batch, batch_idx)
        self.log('pcme_train_loss', pcme_loss, batch_size=len(batch), prog_bar=True, sync_dist=True)
        self.log('train_kld_loss', pcme_dict['vib_loss'], batch_size=len(batch), prog_bar=True, sync_dist=True)
        return pcme_loss

    def validation_step(self, batch, batch_idx):
        pcme_loss, pcme_dict = self(batch, batch_idx)
        self.log('pcme_val_loss', pcme_loss, batch_size=len(batch), prog_bar=True, sync_dist=True)
        self.log('val_kld_loss', pcme_dict['vib_loss'], batch_size=len(batch), prog_bar=True, sync_dist=True)
        return pcme_loss

def get_args():
    parser = ArgumentParser()
    #dataloader arguments
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--dataset_type', type=str, default='h5')
    parser.add_argument('--batch_size',type=int, default=512)
    parser.add_argument('--data_root', type=str, default='/home/a.dhakal/active/proj_smart/satclip_sentinel/images')

    #trainer arguments
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--strategy', type=str, default='ddp_find_unused_parameters_false')
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', default=1)
    parser.add_argument('--mode', type=str, default='dev')
    parser.add_argument('--accumulate_grad', type=int, default=16)

    #logger arguments
    parser.add_argument('--log_dir', type=str, default='/scratch/a.dhakal/hyper_satclip/logs')
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--ckpt_mode', type=str, default='hard')
    parser.add_argument('--project_name', type=str, default='SAPCLIP')
    parser.add_argument('--run_name', type=str, default='dev')
    parser.add_argument('--wandb_mode', type=str, default='disabled')
    parser.add_argument('--wandb_resume', type=str, default='')

    #model arguments
    parser.add_argument('--loss_type', type=str, default='probablistic')
    parser.add_argument('--anneal_T', type=int, default=370)
    parser.add_argument('--contrastive_wt', type=float, default=1.0)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--vision_encoder', type=str, default='CLIP')

    args = parser.parse_args()
    return args

    
if __name__ == '__main__':
    args = get_args()

    #initialize checkpoints and loggers
    lr_logger = LearningRateMonitor(logging_interval='epoch')

    #initiallize the wandb logger
    wb_logger = WandbLogger(save_dir=args.log_dir,project=args.project_name, name=args.run_name,
     mode=args.wandb_mode)

    #initialize checkpoint monitor
    ckpt_monitors = ModelCheckpoint(monitor='val_loss', filename='{epoch}-{val_loss:.3f}',
             mode='min', save_top_k=10, save_last=True)
    
    #initialize trainer
    if args.mode == 'dev': 
        print('Development Test Run')
        trainer = L.Trainer(fast_dev_run=15, max_epochs=4, logger=wb_logger, strategy=args.strategy,
         num_sanity_val_steps=0, accelerator=args.accelerator, devices=args.devices, 
        callbacks=[ckpt_monitors, lr_logger])
    elif args.mode == 'train':
        print('Training Run')
        trainer = L.Trainer(precision='32', max_epochs=args.max_epochs, logger=wb_logger, strategy=args.strategy, 
        num_sanity_val_steps=1, accelerator=args.accelerator, devices=args.devices, 
        callbacks=[ckpt_monitors, lr_logger], check_val_every_n_epoch=1, 
        log_every_n_steps=2, accumulate_grad_batches=args.accumulate_grad)
    else:
        raise ValueError('Invalid value for mode')
    
    #get dataloaders
    if args.dataset_type=='normal':
        dataset = SAPCLIP_Dataset(root=args.data_root, transform_type='sapclip', crop_size=args.crop_size, prototype=False)
    elif args.dataset_type=='h5':
        dataset = SAPCLIP_Dataset_H5(input_path=args.data_root, transform_type='sapclip', crop_size=args.crop_size)

    train_loader, val_loader = get_split_dataset(dataset, val_split=0.05, batch_size=args.batch_size,
     num_workers=args.num_workers)
    print('DataLoaders Initialized')

    #initialize model
    if args.loss_type=='pcme':
        sapclip_model = SAPCLIP_PCME(embed_dim=args.embed_dim, loss_type=args.loss_type,
    anneal_T=args.anneal_T)
    else:
        sapclip_model = SAPCLIP(embed_dim=args.embed_dim, loss_type=args.loss_type,
    anneal_T=args.anneal_T)
    print('SAPCLIP Model Initialized')
    # import code; code.interact(local=dict(globals(), **locals()))
    print('Starting Fit!!!!')
    trainer.fit(sapclip_model, train_dataloaders=train_loader, val_dataloaders=val_loader)








    

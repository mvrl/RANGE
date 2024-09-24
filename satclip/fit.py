import argparse
import os
from datetime import datetime
from argparse import ArgumentParser, Namespace

import lightning.pytorch as L
import torch
import torch.nn.functional as F
# from datamodules.s2geo_dataset import S2GeoDataModule
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.cli import LightningCLI

from sklearn.preprocessing import MinMaxScaler
from einops import repeat
from sklearn.linear_model import RidgeClassifierCV
import pandas as pd
import numpy as np
from rtdl_num_embeddings import PiecewiseLinearEncoding

#local imports 
from .model import SatCLIP_2
from .datamodules.sapclip_dataset import SAPCLIP_Dataset, get_split_dataset, SAPCLIP_Dataset_H5

torch.set_float32_matmul_precision('high')

def str_to_frac(vals):
    vals = vals.split(',')
    l = []
    for val in vals:
        num, denom = val.split('/')
        l.append(int(num)/int(denom))
    return l

def contrastive_loss(similarity, labels):
    #labels is class probablities of shape [N,C], where C is the number of classes 
    return F.cross_entropy(similarity, labels)

# class SAPCLIP(L.LightningModule):
#     def __init__(
#         self,
#         embed_dim=512,
#         image_resolution=256,
#         vision_layers='CLIP',
#         vision_width=768,
#         vision_patch_size=32,
#         in_channels=4,
#         le_type="grid",
#         pe_type="siren",
#         frequency_num=16,
#         max_radius=260,
#         min_radius=1,
#         legendre_polys=16,
#         harmonics_calculation="analytic",
#         sh_embedding_dims=32,
#         learning_rate=1e-4,
#         weight_decay=0.01,
#         num_hidden_layers=2,
#         capacity=256,        
#         loss_type='probablistic',
#        anneal_T=0.001
#     ) -> None:
#         super().__init__()

#         self.model = SatCLIP_2(
#             embed_dim=embed_dim,
#             image_resolution=image_resolution,
#             vision_layers=vision_layers,
#             vision_width=vision_width,
#             vision_patch_size=vision_patch_size,
#             in_channels=in_channels,
#             le_type=le_type,
#             pe_type=pe_type,
#             frequency_num=frequency_num,
#             max_radius=max_radius,
#             min_radius=min_radius,
#             legendre_polys=legendre_polys,
#             harmonics_calculation=harmonics_calculation,
#             sh_embedding_dims=sh_embedding_dims,
#             num_hidden_layers=num_hidden_layers,
#             capacity=capacity,
#             device=self.device,
#             loss_type=loss_type,
#         )
        
#         self.learning_rate = learning_rate
#         self.weight_decay = weight_decay
#         self.anneal_T = anneal_T
#         self.delta_beta = 2/self.anneal_T
#         self.kld_wt=0
#         self.save_hyperparameters()

#     def anneal_beta(self):
#         if self.global_step%self.anneal_T == 0:
#             self.kld_wt = 0
#         elif self.global_step%self.anneal_T>self.anneal_T/2:
#             self.kld_wt = 1
#         else:
#             self.kld_wt+=self.delta_beta
    
#     #increase upto 0.5 and drop back to 0
#     def anneal_beta_2(self):
#         if self.kld_wt>=0.35:
#             self.kld_wt=0     
#         else:
#             self.kld_wt+=self.delta_beta

#     #configure optimizers 
#     def configure_optimizers(self):
#         exclude = (
#             lambda n, p: p.ndim < 2
#             or "bn" in n
#             or "ln" in n
#             or "bias" in n
#             or "logit_scale" in n
#         )
#         include = lambda n, p: not exclude(n, p)

#         named_parameters = list(self.model.named_parameters())
#         gain_or_bias_params = [
#             p for n, p in named_parameters if exclude(n, p) and p.requires_grad
#         ]
#         rest_params = [
#             p for n, p in named_parameters if include(n, p) and p.requires_grad
#         ]

#         optimizer = torch.optim.AdamW(
#             [
#                 {"params": gain_or_bias_params, "weight_decay": 0.0},
#                 {
#                     "params": rest_params,
#                     "weight_decay": self.weight_decay,
#                 },  # specify in configs/default.yaml
#             ],
#             lr=self.learning_rate,  # specify in configs/default.yaml
#         )
#         return optimizer
    
#     #define the forward pass
#     def forward(self, batch, batch_idx):       
#         contrastive_loss, kld_loss = self.model(batch)

#         return contrastive_loss, kld_loss

#     def training_step(self, batch, batch_idx):
#         contrastive_loss, kld_loss = self(batch, batch_idx)
#         loss = contrastive_loss + self.kld_wt * kld_loss
#         self.anneal_beta_2()
#         self.log('train_contrastive_loss', contrastive_loss, batch_size=len(batch), prog_bar=True, sync_dist=True)
#         self.log('train_kld_loss', kld_loss, batch_size=len(batch), prog_bar=True, sync_dist=True)
#         self.log('train_loss', loss, batch_size=len(batch), prog_bar=True, sync_dist=True)
#         self.log('beta', self.kld_wt, sync_dist=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         contrastive_loss, kld_loss = self(batch, batch_idx)
#         loss = contrastive_loss + kld_loss
#         self.log('val_contrastive_loss', contrastive_loss, batch_size=len(batch), prog_bar=True, sync_dist=True)
#         self.log('val_kld_loss', kld_loss, batch_size=len(batch), prog_bar=True, sync_dist=True)
#         self.log('val_loss', loss, batch_size=len(batch), prog_bar=True, sync_dist=True)
#         return loss    

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
        weight_decay=0.02,
        num_hidden_layers=2,
        capacity=256,        
        loss_type='pcme',
        anneal_T=1000,
        scale_encoding='onehot',
        scale_bins=3,
        **kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
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
            loss_type=loss_type,
            scale_encoding=scale_encoding,
            scale_bins=scale_bins,
            **kwargs
        )
        self.loss_type=loss_type
        self.scale_encoding = scale_encoding
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.anneal_T = anneal_T
        self.delta_beta = 2/self.anneal_T
        self.kld_wt=0
        self.warmup_steps = kwargs.get('warmup_steps')
        bins =  [torch.from_numpy(np.linspace(0,6,scale_bins+1)).double()]
        
        
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
        #filter trainable parameters
        params = list(filter(lambda p:p.requires_grad, self.parameters()))
        self.optimizer = torch.optim.AdamW(params,
         lr=self.learning_rate,
         weight_decay=self.weight_decay,
         betas=(0.9, 0.999),
         eps=1e-6)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=self.warmup_steps)
    
        return {'optimizer': self.optimizer, 
        'lr_scheduler': {
            'name':'train/lr',
            'scheduler': self.scheduler,
            'interval': 'step',
            'frequency': 1
        }
        }

    
    #define the forward pass
    def forward(self, batch):      
        loss, loss_dict = self.model(batch)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self(batch)
        if self.loss_type=='clip':
            self.log('train_loss', loss, batch_size=len(batch), prog_bar=True, sync_dist=True)
            self.log('logit_scale', loss_dict['logit_scale'], batch_size=len(batch), prog_bar=True, sync_dist=True)
        else:
            self.log('train_loss', loss, batch_size=len(batch), prog_bar=True, sync_dist=True)
            self.log('train_kld_loss', loss_dict['vib_loss'], batch_size=len(batch), prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self(batch)
        if self.loss_type=='clip':
            self.log('val_loss', loss, batch_size=len(batch), prog_bar=True, sync_dist=True)
            self.log('logit_scale', loss_dict['logit_scale'], batch_size=len(batch), prog_bar=True, sync_dist=True)
        else:
            self.log('val_loss', loss, batch_size=len(batch), prog_bar=True, sync_dist=True)
            self.log('val_kld_loss', loss_dict['vib_loss'], batch_size=len(batch), prog_bar=True, sync_dist=True)
        return loss
    
    def on_validation_epoch_end(self):
       # only log every 3rd epoch
        if self.current_epoch%5==0:
             # read the data
            biome_df = pd.read_csv('/projects/bdec/adhakal2/hyper_satclip/data/eval_data/biome_train_new.csv')
            eco_df = pd.read_csv('/projects/bdec/adhakal2/hyper_satclip/data/eval_data/ecoregion_train_new.csv')

            # df = pd.concat([df_train, df_test])
            labels_biome = biome_df['BIOME_NAME']
            labels_eco = eco_df['ECO_NAME']
            # ytrain_biome = labels_biome[:len(df_train)]
            # ytest_biome = labels_biome[len(df_train):]

            # y_train_eco = labels_eco[:len(df_train)]
            # y_test_eco = labels_eco[len(df_train):]

            biome_coords = biome_df[['X', 'Y']].values
            biome_coords = torch.from_numpy(biome_coords).double()
            biome_loc = biome_coords.to('cuda')

            eco_coords = eco_df[['X', 'Y']].values
            eco_coords = torch.from_numpy(eco_coords).double()
            eco_loc = eco_coords.to('cuda')

            #get the model
            sapclip_encoder = self.model.eval()
            
            #compute embeddings at each scale
            if self.scale_encoding=='learnable':
                self.map_scale = {1:torch.tensor(0).cuda(),
                3:torch.tensor(1).cuda(), 
                5:torch.tensor(2).cuda()}
                scale_1 = self.map_scale[1]
                scale_1 = scale_1.repeat(len(loc)) #scale_1 = repeat(scale_1, 'd -> b d', b=len(loc)).cuda()
                scale_3 = self.map_scale[3]
                scale_3 = scale_3.repeat(len(loc)) #scale_3 = repeat(scale_3, 'd -> b d', b=len(loc)).cuda()
                scale_5 = self.map_scale[5]
                scale_5 = scale_5.repeat(len(loc)) #scale_5 = repeat(scale_5, 'd -> b d', b=len(loc)).cuda() 
            elif self.scale_encoding=='onehot':
                self.map_scale = {1:torch.tensor([1,0,0]),3:torch.tensor([0,1,0]),5:torch.tensor([0,0,1])}
                scale_1 = self.map_scale[1]
                biome_scale_1 = repeat(scale_1, 'd -> b d', b=len(biome_loc)).cuda()
                eco_scale_1 = repeat(scale_1, 'd -> b d', b=len(eco_loc)).cuda()
                scale_3 = self.map_scale[3]
                biome_scale_3 = repeat(scale_3, 'd -> b d', b=len(biome_loc)).cuda()
                eco_scale_3 = repeat(scale_3, 'd -> b d', b=len(eco_loc)).cuda()
                scale_5 = self.map_scale[5]
                biome_scale_5 = repeat(scale_5, 'd -> b d', b=len(biome_loc)).cuda()
                eco_scale_5 = repeat(scale_5, 'd -> b d', b=len(eco_loc)).cuda()

            # generate sapclip embeddings
            #instead of computing sapclip embeddings for individual scale, compute for all scales and average the embeddings
            if 'pcme' in self.loss_type:
                pass
                # loc_embeddings = sapclip_encoder.encode_location(coords=loc, hot_scale=scale_1)
                # loc_mu_1 = loc_embeddings[0].detach().cpu().numpy()
                # loc_sigma_1 = torch.exp(loc_embeddings[1]).detach().cpu().numpy()
                # loc_embeddings = sapclip_encoder.encode_location(coords=loc, hot_scale=scale_3)
                # loc_mu_3 = loc_embeddings[0].detach().cpu().numpy()
                # loc_sigma_3 = torch.exp(loc_embeddings[1]).detach().cpu().numpy()
                # loc_embeddings = sapclip_encoder.encode_location(coords=loc, hot_scale=scale_5)
                # loc_mu_5 = loc_embeddings[0].detach().cpu().numpy()
                # loc_sigma_5 = torch.exp(loc_embeddings[1]).detach().cpu().numpy()

            elif 'clip' in self.loss_type:
                #biome embeddings
                loc_mu_1 = sapclip_encoder.encode_location(coords=biome_loc, hot_scale=biome_scale_1).detach().cpu().numpy()
                loc_mu_3 = sapclip_encoder.encode_location(coords=biome_loc, hot_scale=biome_scale_3).detach().cpu().numpy()
                loc_mu_5 = sapclip_encoder.encode_location(coords=biome_loc, hot_scale=biome_scale_5).detach().cpu().numpy()
                loc_mu_all = {1:loc_mu_1, 3:loc_mu_3, 5:loc_mu_5}            
                #biome accuracy
                for scale in loc_mu_all:
                    loc_mu = loc_mu_all[scale]
                    xtrain_sapclip = loc_mu
                    scaler = MinMaxScaler()
                    xtrain_sapclip = scaler.fit_transform(xtrain_sapclip)
                    
                    clf_biome = RidgeClassifierCV(alphas=(0.1, 1.0, 10.0), cv=10)
                    clf_biome.fit(xtrain_sapclip, labels_biome)
                    acc_biome = clf_biome.score(xtrain_sapclip, labels_biome)
                    self.log(f'BIOME_ACC_{scale}', acc_biome, prog_bar=True, sync_dist=True)
                #eco embeddings
                loc_mu_1 = sapclip_encoder.encode_location(coords=eco_loc, hot_scale=eco_scale_1).detach().cpu().numpy()
                loc_mu_3 = sapclip_encoder.encode_location(coords=eco_loc, hot_scale=eco_scale_3).detach().cpu().numpy()
                loc_mu_5 = sapclip_encoder.encode_location(coords=eco_loc, hot_scale=eco_scale_5).detach().cpu().numpy()
                loc_mu_all = {1:loc_mu_1, 3:loc_mu_3, 5:loc_mu_5}            
                #biome accuracy
                for scale in loc_mu_all:
                    loc_mu = loc_mu_all[scale]
                    xtrain_sapclip = loc_mu
                    scaler = MinMaxScaler()
                    xtrain_sapclip = scaler.fit_transform(xtrain_sapclip)
                    
                    clf_biome = RidgeClassifierCV(alphas=(0.1, 1.0, 10.0), cv=10)
                    clf_biome.fit(xtrain_sapclip, labels_eco)
                    acc_biome = clf_biome.score(xtrain_sapclip, labels_eco)
                    self.log(f'ECO_ACC_{scale}', acc_biome, prog_bar=True, sync_dist=True)

                
        else:
            pass


def get_args():
    parser = ArgumentParser()
    #dataloader arguments
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--dataset_type', type=str, default='h5')
    parser.add_argument('--batch_size',type=int, default=512)
    parser.add_argument('--data_root', type=str, default='/home/a.dhakal/active/proj_smart/satclip_sentinel/images')
    parser.add_argument('--scale_ratio', type=str, default='1/3,1/3,1/3')
    parser.add_argument('--crop_type', type=str, choices=['resized', 'sampled'], help='Type of crop')

    #trainer arguments
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--strategy', type=str, default='ddp_find_unused_parameters_false')
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mode', type=str, default='dev')
    parser.add_argument('--accumulate_grad', type=int, default=16)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    #logger arguments
    parser.add_argument('--log_dir', type=str, default='/scratch/a.dhakal/hyper_satclip/logs')
    parser.add_argument('--ckpt_mode', type=str, default='hard')
    parser.add_argument('--project_name', type=str, default='SAPCLIP')
    parser.add_argument('--run_name', type=str, default='dev')
    parser.add_argument('--wandb_mode', type=str, default='disabled')
    parser.add_argument('--wandb_resume', type=str, default='')
    parser.add_argument('--wandb_notes', type=str, default='Description of the model here')


    #model arguments
    parser.add_argument('--scale_encoding', type=str, default='onehot', choices=['onehot', 'ple', 'learnable'])
    parser.add_argument('--scale_bins', type=int, default=50)
    parser.add_argument('--transform_type', type=str, default='sapclip')
    parser.add_argument('--loss_type', type=str, default='probablistic')
    parser.add_argument('--sampling_num', type=int, default=10)
    parser.add_argument('--anneal_T', type=int, default=370)
    parser.add_argument('--contrastive_wt', type=float, default=1.0)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--vision_encoder', type=str, default='CLIP')
    parser.add_argument('--pretrained_satclip', action='store_true', default=False)
    parser.add_argument('--early_fusion', action='store_true', default=False)
    parser.add_argument('--num_transformer_layers', type=int, default=3)

    args = parser.parse_args()
    return args

    
if __name__ == '__main__':
    args = get_args()

    #initialize checkpoints and loggers
    lr_logger = LearningRateMonitor(logging_interval='step')

    #initiallize the wandb logger
    wb_logger = WandbLogger(save_dir=args.log_dir,project=args.project_name, name=args.run_name,
     mode=args.wandb_mode, notes=args.wandb_notes)
    #initialize checkpoint monitor
    ckpt_monitors = (ModelCheckpoint(monitor='val_loss', filename='{epoch}-{val_loss:.3f}',
             mode='min', save_top_k=10, save_last=True),
             ModelCheckpoint(monitor='BIOME_ACC_1', filename='{epoch}-{acc_biome_1:.3f}',
             mode='max', save_top_k=5, save_last=True),
             ModelCheckpoint(monitor='BIOME_ACC_3', filename='{epoch}-{acc_biome_3:.3f}',
             mode='max', save_top_k=5, save_last=True),
             ModelCheckpoint(monitor='BIOME_ACC_5', filename='{epoch}-{acc_biome_5:.3f}',
             mode='max', save_top_k=5, save_last=True),
             ModelCheckpoint(monitor='ECO_ACC_1', filename='{epoch}-{acc_eco_1:.3f}',
             mode='max', save_top_k=5, save_last=True),
             ModelCheckpoint(monitor='ECO_ACC_3', filename='{epoch}-{acc_eco_3:.3f}',
             mode='max', save_top_k=5, save_last=True),
             ModelCheckpoint(monitor='ECO_ACC_5', filename='{epoch}-{acc_eco_5:.3f}',
             mode='max', save_top_k=5, save_last=True)
             )

    #initialize trainer
    if args.mode == 'dev': 
        print('Development Test Run')
        trainer = L.Trainer(fast_dev_run=15, max_epochs=4, logger=wb_logger, strategy=args.strategy,
         num_sanity_val_steps=0, accelerator=args.accelerator, devices=args.devices, 
        callbacks=[*ckpt_monitors, lr_logger])
    elif args.mode == 'train':
        print('Training Run')
        trainer = L.Trainer(precision='64', max_epochs=args.max_epochs, logger=wb_logger, strategy=args.strategy, 
        num_sanity_val_steps=1, accelerator=args.accelerator, devices=args.devices, 
        callbacks=[*ckpt_monitors, lr_logger], check_val_every_n_epoch=1, 
        log_every_n_steps=1, accumulate_grad_batches=args.accumulate_grad)
    else:
        raise ValueError('Invalid value for mode')
    
    #get dataloaders
    
    args.scale_ratio = str_to_frac(args.scale_ratio)
    assert args.scale_ratio[0] + args.scale_ratio[1] + args.scale_ratio[2] == 1.0, 'Scale ratio should sum to 1'
    
    #initialize dataset
    dataset = SAPCLIP_Dataset(root=args.data_root, transform_type=args.transform_type,
     crop_size=args.crop_size, prototype=False, scale_encoding=args.scale_encoding,
      scale_bins=args.scale_bins, scale_ratio=args.scale_ratio,
      crop_type=args.crop_type)
    #initailize dataloaders
    train_loader, val_loader = get_split_dataset(dataset, val_split=0.05, batch_size=args.batch_size,
     num_workers=args.num_workers, transform_type=args.transform_type)
    print('DataLoaders Initialized')
    #initialize model
    #use the more recent SAPCLIP_PCME if loss is pcme or clip
    if 'pcme' in args.loss_type or 'clip' in args.loss_type:
        if args.ckpt_path:
            print('Starting training from ckpt')
            SAPCLIP_PCME.load_from_checkpoint(args.ckpt_path)
        else:
            print('Starting fresh training')
            sapclip_model = SAPCLIP_PCME(embed_dim=args.embed_dim, loss_type=args.loss_type,
    vision_layers=args.vision_encoder, anneal_T=args.anneal_T,
    scale_encoding=args.scale_encoding, scale_bins=args.scale_bins,
    satclip_pretrained=args.pretrained_satclip, early_fusion=args.early_fusion,
    num_t_layers=args.num_transformer_layers, 
    device=args.device,warmup_steps=args.warmup_steps)
    #use the original PCME with probablistic loss
    elif args.loss_type=='probablistic':
        print('Using likelihood type loss')
        sapclip_model = SAPCLIP(embed_dim=args.embed_dim, loss_type=args.loss_type,
    anneal_T=args.anneal_T)

    print('SAPCLIP Model Initialized')
    # import code; code.interact(local=dict(globals(), **locals()))
    print('Starting Fit!!!!')
    trainer.fit(sapclip_model, train_dataloaders=train_loader, val_dataloaders=val_loader)








    

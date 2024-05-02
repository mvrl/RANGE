import argparse
import os
from datetime import datetime

import lightning.pytorch
import torch
# from datamodules.s2geo_dataset import S2GeoDataModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI


#local imports 
from .loss import SAPCLIPLoss
from .model import SatCLIP_2
from .datamodules.sapclip_dataset import SAPCLIP_Dataset, get_split_dataset

torch.set_float32_matmul_precision('high')



class SAPCLIP(lightning.pytorch.LightningModule):
    def __init__(
        self,
        embed_dim=512,
        image_resolution=256,
        vision_layers='SatMAE',
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
            device=self.device
        )
        
        self.loss_fun = SAPCLIPLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters()

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
        
        logits_per_image, logits_per_coord = self.model(batch['image'], batch['point'], batch['hot_scale'])
        return self.loss_fun(logits_per_image, logits_per_coord)

    def training_step(self, batch, batch_idx):
        loss = self(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch, batch_idx)
        self.log("val_loss", loss)
        return loss    
    
if __name__ == '__main__':
    root = '/home/a.dhakal/active/proj_smart/satclip_sentinel/images'

    #get dataloaders
    dataset = SAPCLIP_Dataset(root=root,transform_type='sapclip', crop_size=224, prototype=True)
    train_loader, val_loader = get_split_dataset(dataset, val_split=0.1, batch_size=4, num_workers=0)
    
    sample = next(iter(val_loader))

    #initialize model
    sapclip_model = SAPCLIP()

    #forward pass
    output = sapclip_model(sample, 0)
    








    

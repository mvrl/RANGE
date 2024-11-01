import torch
import torch.nn as nn
import lightning.pytorch as L
from lightning.pytorch.loggers import WandbLogger

from torch.utils.data import DataLoader,Dataset
from torch.utils.data._utils.collate import default_collate
from torchmetrics import Accuracy

import numpy as np
import os
import glob
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm
import argparse
#local import
from .load import get_satclip
from .utils.make_lc import LCProb

class DbDataset(Dataset):
    def __init__(self, db_path):
        ranf_db = np.load(db_path, allow_pickle=True)
        self.db_locs_latlon = ranf_db['locs']
        #grab the latlon
        self.db_locs_latlon = torch.from_numpy(self.db_locs_latlon).double()
        #grab the low res satclip embeddings
        self.db_satclip_embeddings = ranf_db['satclip_embeddings']
        self.db_satclip_embeddings = torch.from_numpy(self.db_satclip_embeddings).double()
        #grab the high res embeddings
        #normalize the embeddings
        self.db_high_resolution_satclip_embeddings = torch.tensor(ranf_db['image_embeddings']).double()


    def __len__(self):
        return len(self.db_locs_latlon)
    
    def __getitem__(self, idx):
        return self.db_locs_latlon[idx], self.db_satclip_embeddings[idx], self.db_high_resolution_satclip_embeddings[idx], idx

def collate_fn_remove_none(batch):
    # Filter out None values from the batch
    batch = [item for item in batch if item is not None]
    # If the batch is empty after filtering, return an empty list
    if not batch:
        return []
    # Use the default collate_fn to combine the remaining items
    return default_collate(batch)

class LandcoverDataset(Dataset):
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.imgs = glob.glob(self.dir_path+'/*.jpeg')
        self.lc_prob = LCProb()

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        #read the image
        image = Image.open(img_path).convert('RGB')
        #get the dictionary of probabilities
        labels = self.get_landcover_labels(image)
        #convert them to tensor
        labels = torch.tensor(list(labels.values())).double()
        #grab the lonlat from the image path name
        lon_lat_str = (img_path.split('/')[-1]).replace('.jpeg','').split('_')[1:3]
        idx_number = int((img_path.split('/')[-1]).split('_')[0])
        lon_lat = torch.tensor([float(l) for l in lon_lat_str]).double()
        if torch.argmax(labels) ==0:
            return None
        return torch.tensor([idx_number]), lon_lat, labels 

    #convert landcover image to probabilities 
    def get_landcover_labels(self, img):
        img = np.array(img)
        img = self.lc_prob.discretize_img(img)
        return self.lc_prob.im_to_prob(img)

class LandcoverNPZDataset(Dataset):
    def __init__(self, npz_path):
        self.npz_path = npz_path
        data = np.load(npz_path)
        self.lonlats = data['lonlats']
        self.probs = data['landcover_probs']
        self.indices = data['index']
        
    def __getitem__(self, idx):
        return torch.tensor(self.indices[idx]), torch.tensor(self.lonlats[idx]).double(), torch.tensor(self.probs[idx]).double()

    def __len__(self):
        return len(self.indices)
    
# model for training

class temp_layer(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(temperature))

#l2 normalize torch embeddings
def normalize(x):
    return x / x.norm(p=2, dim=-1, keepdim=True)

class TempModel(L.LightningModule):
    def __init__(self,args,  **kwargs):
        super().__init__()
        #get batch_size
        self.temp_init = args.temp_init
        self.lr = args.lr
        self.wt_decay = args.wt_decay
        self.num_classes = args.num_classes
        #get satcilp location encoder
        self.loc_model = get_satclip(
                    hf_hub_download("microsoft/SatCLIP-ViT16-L40", "satclip-vit16-l40.ckpt", force_download=False),
                device=self.device).double()
        for param in self.loc_model.parameters():
            param.requires_grad = False
        #grab the ranf_db
        self.db_path = args.db_path
        ranf_db = np.load(self.db_path, allow_pickle=True)
        self.db_locs_latlon = torch.tensor(ranf_db['locs'].astype(np.float64))
        self.db_satclip_embeddings = torch.tensor(ranf_db['satclip_embeddings'].astype(np.float64))
        self.db_high_resolution_satclip_embeddings = torch.tensor(ranf_db['image_embeddings'].astype(np.float64))
        
        #initialize the temperature
        self.temp_layer = temp_layer(self.temp_init)
        #initialize the linear layer
        self.inp_size = self.db_satclip_embeddings.shape[-1] + self.db_high_resolution_satclip_embeddings.shape[-1]
        self.linear = nn.Linear(self.inp_size, self.num_classes).double()

        #initialize the loss
        self.loss = nn.CrossEntropyLoss()
        #initialize the accuracy
        self.accuracy = Accuracy(task='multiclass', num_classes=self.num_classes)

    def forward(self, batch):
        idx, loc, prob = batch
        #get the satclip embeddings for the locations
        loc_embeddings = self.loc_model(loc)
        #compute similarity with db images
        similarity = normalize(loc_embeddings) @ normalize(self.db_satclip_embeddings.T).to(self.device)   
        similarity[torch.arange(similarity.shape[0]), idx] = -1e10
        #scale the similarities
        logit_scale = self.temp_layer.logit_scale.exp()
        similarity = similarity * logit_scale
        #convert to probabilities
        similarity = similarity.softmax(dim=-1)
        #average the high resolution embeddings using similarities
        averaged_high_res_embeddings = similarity @ self.db_high_resolution_satclip_embeddings.to(self.device)
        #concatenate the embeddings
        range_embeddings = torch.cat([loc_embeddings, averaged_high_res_embeddings], dim=-1)
        #forward through the linear layer
        logits = self.linear(range_embeddings)
        #get the class with max probability
        prob = torch.argmax(prob, dim=-1)
        #compute cross entropy loss
        loss = self.loss(logits, prob)
        #compute accuracy
        acc = self.accuracy(logits, prob)

        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch)
        self.log('train_loss', loss, prog_bar=True)
        self.log('tain_acc', acc, prog_bar=True)
        self.log('temperature', self.temp_layer.logit_scale.exp().item(), prog_bar=True) 
        return loss

    def validation_step(self, batch, batch_idx):
        loss,acc = self.forward(batch)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        self.params = list(filter(lambda p: p.requires_grad, self.parameters()))
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr, weight_decay=self.wt_decay)
        return self.optimizer
    
def get_args():
    parser = argparse.ArgumentParser(description='Train the temperature model')
    parser.add_argument('--run_type', type=str, help='train or dev', default='train')
    parser.add_argument('--landcover_npz_path', type=str, help='path to the landcover npz file', default='/projects/bdec/adhakal2/hyper_satclip/data/landcover_data/images_corrected/land_cover.npz')
    parser.add_argument('--temp_init', type=int, help='initial temperature', default=20)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-04)
    parser.add_argument('--max_epochs', type=int, help='maximum number of epochs', default=1000)
    parser.add_argument('--db_path', type=str, help='path to the ranf db', default='/projects/bdec/adhakal2/hyper_satclip/data/models/ranf/ranf_satmae_db.npz')
    parser.add_argument('--wt_decay', type=float, help='weight decay', default=1e-4)
    parser.add_argument('--num_classes', type=int, help='number of classes', default=12)
    return parser.parse_args()

if __name__ == '__main__':
    #create model
    create_prob = False
    if create_prob:
        img_dir = '/projects/bdec/adhakal2/hyper_satclip/data/landcover_data/images_corrected/land_cover'
        dataset = LandcoverDataset(img_dir)
        dataloader = DataLoader(dataset, batch_size=100, num_workers=30, shuffle=False, collate_fn=collate_fn_remove_none, drop_last=False)
        lonlats = []
        probs = []
        indices = []
        for i, data in tqdm(enumerate(dataloader)):
            idx, lonlat, prob = data
            lonlats.append(lonlat)
            probs.append(prob)
            indices.append(idx)
        probs = torch.cat(probs,dim=0).numpy()
        lonlats = torch.cat(lonlats,dim=0).numpy()
        indices = torch.cat(indices,dim=0).numpy()
    
        
        np.savez('/projects/bdec/adhakal2/hyper_satclip/data/landcover_data/images_corrected/land_cover.npz', lonlats=lonlats, landcover_probs=probs, index=indices)
        print('Saved')
            
    else:
        args = get_args()
        #create dataset and dataloader
        dataset = LandcoverNPZDataset(args.landcover_npz_path)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])
        train_dataloader = DataLoader(train_dataset, batch_size=10000, num_workers=4, shuffle=True, drop_last=False)
        val_dataloader = DataLoader(val_dataset, batch_size=10000, num_workers=4, shuffle=False, drop_last=False)

        #create the logger
        run_name = f'temp_{args.temp_init}_Adam_{args.lr}_wtdecay_{args.wt_decay}'
        wb_logger = WandbLogger(save_dir='/projects/bdec/adhakal2/hyper_satclip/logs', project='RAN-GE', name=run_name, mode='online')
 
        #initialize the model
        model = TempModel(args)
        
        #create trainer
        if args.run_type == 'train':
            trainer = L.Trainer(precision='32', max_epochs=args.max_epochs, logger=wb_logger, strategy='ddp_find_unused_parameters_false', num_sanity_val_steps=1,
                        accelerator='gpu', devices=1, check_val_every_n_epoch=1, log_every_n_steps=5)
        elif args.run_type == 'dev':
            trainer = L.Trainer(fast_dev_run=10, precision='32', max_epochs=args.max_epochs, logger=wb_logger, strategy='ddp_find_unused_parameters_false', num_sanity_val_steps=1,
                        accelerator='gpu', devices=1, check_val_every_n_epoch=1, log_every_n_steps=5)

        #train the model
        trainer.fit(model, train_dataloader, val_dataloader)
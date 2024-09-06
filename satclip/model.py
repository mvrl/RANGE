from collections import OrderedDict
from typing import Tuple, Union, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math
import os

import timm
import torchgeo.models
from torchgeo.models import ResNet18_Weights, ResNet50_Weights, ViTSmall16_Weights
from huggingface_hub import hf_hub_download


from transformers import CLIPVisionModelWithProjection


#local imports
from .location_encoder import get_positional_encoding, get_neural_network, LocationEncoder
# from .datamodules.s2geo_dataset import S2Geo
from .datamodules.sapclip_dataset import SAPCLIP_Dataset, get_split_dataset, SAPCLIP_Dataset_H5
from .vision_models.clip import Clip
from .vision_models.satmae import SatMAE
from .vision_models.temp import temp_layer
from .loss.pcme import MCSoftContrastiveLoss
from .utils.utils import sample_gaussian_tensors
from .load import get_satclip



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


# #original SatCLIP
# class SatCLIP(nn.Module):
#     def __init__(self,
#                  embed_dim: int,
#                  # vision
#                  image_resolution: int,
#                  vision_layers: Union[Tuple[int, int, int, int], int, str],
#                  vision_width: int,
#                  vision_patch_size: int,
#                  in_channels: int,
#                  # location
#                  le_type: str,
#                  pe_type: str,
#                  frequency_num: int, 
#                  max_radius: int,  
#                  min_radius: int,
#                  harmonics_calculation: str,
#                  legendre_polys: int=10, 
#                  sh_embedding_dims: int=16, 
#                  ffn: bool=True,
#                  num_hidden_layers: int=2,
#                  capacity: int=256,
#                  *args,
#                  **kwargs
#                  ):
#         super().__init__()
            
#         if isinstance(vision_layers, (tuple, list)):
#             print('using modified resnet')
#             vision_heads = vision_width * 32 // 64
#             self.visual = ModifiedResNet(
#                 layers=vision_layers,
#                 output_dim=embed_dim,
#                 heads=vision_heads,
#                 input_resolution=image_resolution,
#                 width=vision_width,
#                 in_channels=in_channels
#             )
            
#         elif vision_layers == 'moco_resnet18':
#             print('using pretrained moco resnet18')
#             weights = ResNet18_Weights.SENTINEL2_ALL_MOCO
#             in_chans = weights.meta["in_chans"]
#             self.visual = timm.create_model("resnet18", in_chans=in_chans, num_classes=embed_dim)
#             self.visual.load_state_dict(weights.get_state_dict(progress=True), strict=False)
#             self.visual.requires_grad_(False)
#             self.visual.fc.requires_grad_(True)

#         elif vision_layers == 'moco_resnet50':
#             print('using pretrained moco resnet50')
#             weights = ResNet50_Weights.SENTINEL2_ALL_MOCO
#             in_chans = weights.meta["in_chans"]
#             self.visual = timm.create_model("resnet50", in_chans=in_chans, num_classes=embed_dim)
#             self.visual.load_state_dict(weights.get_state_dict(progress=True), strict=False)
#             self.visual.requires_grad_(False)
#             self.visual.fc.requires_grad_(True)
            
#         elif vision_layers == 'moco_vit16':
#             print('using pretrained moco vit16')
#             weights = ViTSmall16_Weights.SENTINEL2_ALL_MOCO
#             in_chans = weights.meta["in_chans"]
#             self.visual = timm.create_model("vit_small_patch16_224", in_chans=in_chans, num_classes=embed_dim)
#             self.visual.load_state_dict(weights.get_state_dict(progress=True), strict=False)
#             self.visual.requires_grad_(False)
#             self.visual.head.requires_grad_(True)
        
#         elif vision_layers == 'SATMAE':
#             print('Using Scale MAE')
#             pretrained_satmae_path = '/home/a.dhakal/active/user_a.dhakal/hyper_satclip/data/satmae_models/pretrain-vit-base-e199.pth'
#             self.visual = SatMAE(pretrained_models_path=pretrained_path, device=device, fc_dim=embed_dim)
#             self.visual.required_grad_(False)
#             self.visual.fc.required_grad_(True)
#             #### need to add SatMAE

#         else:
#             print('using vision transformer')
#             vision_heads = vision_width // 64
#             self.visual = VisionTransformer(
#                 input_resolution=image_resolution,
#                 patch_size=vision_patch_size,
#                 width=vision_width,
#                 layers=vision_layers,
#                 heads=vision_heads,
#                 output_dim=embed_dim,
#                 in_channels=in_channels
#             )
        
#         satclip_pretrained = kwargs.get('satclip_pretrained')
#         if not satclip_pretrained:        
#             print('Initializing new SatCLIP')
#             self.posenc = get_positional_encoding(name=le_type, harmonics_calculation=harmonics_calculation, legendre_polys=legendre_polys, min_radius=min_radius, max_radius=max_radius, frequency_num=frequency_num).double()
#             self.nnet = get_neural_network(name=pe_type, input_dim=self.posenc.embedding_dim, num_classes=embed_dim, dim_hidden=capacity, num_layers=num_hidden_layers).double()
#             self.location = LocationEncoder(self.posenc, 
#                                             self.nnet
#             ).double()
#         else:
#             print('Loading pretrained SatCLIP')
#             self.location = get_satclip(
#                     hf_hub_download("microsoft/SatCLIP-ViT16-L40", "satclip-vit16-l40.ckpt", force_download=False),
#                 device=device).double()
#             self.posenc = self.location.posenc.double()
#             self.nnet = self.location.nnet.double()
        
#         self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
#         self.initialize_parameters()

#     def initialize_parameters(self):
#         if isinstance(self.visual, ModifiedResNet):
#             if self.visual.attnpool is not None:
#                 std = self.visual.attnpool.c_proj.in_features ** -0.5
#                 nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
#                 nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
#                 nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
#                 nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

#             for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
#                 for name, param in resnet_block.named_parameters():
#                     if name.endswith("bn3.weight"):
#                         nn.init.zeros_(param)

#     @property
#     def dtype(self):
#         if isinstance(self.visual, timm.models.vision_transformer.VisionTransformer):
#             return self.visual.patch_embed.proj.weight.dtype
#         else:
#             return self.visual.conv1.weight.dtype

#     def encode_image(self, image):
#         return self.visual(image.type(self.dtype))

#     def encode_location(self, coords):
#         return self.location(coords.double())

#     def forward(self, image, coords):

#         image_features = self.encode_image(image)     
#         location_features = self.encode_location(coords).float()
#         # normalized features
#         image_features = image_features / image_features.norm(dim=1, keepdim=True)
#         location_features = location_features / location_features.norm(dim=1, keepdim=True)

#         # cosine similarity as logits
#         logit_scale = self.logit_scale.exp()
#         logits_per_image = logit_scale * image_features @ location_features.t()
#         logits_per_location = logits_per_image.t()

#         # shape = [global_batch_size, global_batch_size]
#         return logits_per_image, logits_per_location


# def compute_isotropic_kld(mean, std):
#     kld = -1/2*(1+torch.log(std**2)-std**2-mean**2).sum(dim=-1)
#     return kld

class MiniTransformer(nn.Module):
    def __init__(self, input_dims, forward_dims, num_layers, num_heads, num_tokens):
        super(MiniTransformer, self).__init__()
        transformer_layer = nn.TransformerEncoderLayer(d_model=input_dims,
             nhead=num_heads, dim_feedforward=forward_dims, batch_first=True).double()
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers).double() # shape = [batch, n_tokens, dim]
        self.positional_encoding = nn.Parameter(torch.randn(1, num_tokens, input_dims))
        
    def forward(self, x):
        x = x + self.positional_encoding
        output = self.transformer(x)
        output = output.mean(dim=1, keepdim=False)
        return output
        

#my satclip class
class SatCLIP_2(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int, str],
                 vision_width: int,
                 vision_patch_size: int,
                 in_channels: int,
                 # location
                 le_type: str,
                 pe_type: str,
                 frequency_num: int, 
                 max_radius: int,  
                 min_radius: int,
                 harmonics_calculation: str,
                 legendre_polys: int=10, 
                 sh_embedding_dims: int=16, 
                 ffn: bool=True,
                 num_hidden_layers: int=2,
                 capacity: int=256,
                 device: str='cuda',
                 loss_type: str='probablistic',
                 scale_encoding: str='onehot',
                 scale_bins: int=3,
                 *args,
                 **kwargs
                 ):
        super().__init__()

        self.vision_layers = vision_layers
        self.device = device
        self.scale_encoding = scale_encoding
        #define the vision encoder

        # SatMAE encoder        
        if vision_layers == 'SatMAE':
            print('Using Sat MAE')
            self.visual = SatMAE(embed_dim=embed_dim, device=device)
            self.visual.requires_grad_(False)
            for name,param in self.visual.named_parameters():
                if 'projection_layer' in name:
                    param.requires_grad=True
            
        # CLIP encoder
        elif vision_layers == 'CLIP':
            print('Using CLIP')
            self.visual=Clip(embed_dim=embed_dim, device=device, vit_type='16')
            self.visual.requires_grad_(False) 
            for name,param in self.visual.named_parameters():
                if 'projection_layer' in name:
                    param.requires_grad=True
        else:
            print('using vision transformer')
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                in_channels=in_channels
            )
        
        #define the satclip location encoder
        satclip_pretrained = kwargs.get('satclip_pretrained')
        if not satclip_pretrained:        
            print('Initializing new SatCLIP')
            self.posenc = get_positional_encoding(name=le_type, harmonics_calculation=harmonics_calculation, legendre_polys=legendre_polys, min_radius=min_radius, max_radius=max_radius, frequency_num=frequency_num).double()
            self.nnet = get_neural_network(name=pe_type, input_dim=self.posenc.embedding_dim, num_classes=embed_dim, dim_hidden=capacity, num_layers=num_hidden_layers).double()
            self.location = LocationEncoder(self.posenc, 
                                            self.nnet
            ).double()
        else:
            print('Loading pretrained SatCLIP')
            self.location = get_satclip(
                    hf_hub_download("microsoft/SatCLIP-ViT16-L40", "satclip-vit16-l40.ckpt", force_download=False),
                device=device).double()
            self.posenc = self.location.posenc.double()
            self.nnet = self.location.nnet.double()
        
        


        #define the scale encoder
        # self.scale_encoder = nn.Sequential(nn.Linear(scale_bins, embed_dim).double(),
        #                                    nn.LeakyReLU(0.1).double(),
        #                                    nn.Linear(embed_dim, embed_dim).double())
        self.early_fusion = kwargs.get('early_fusion')
        self.num_t_layers = kwargs.get('num_t_layers')
        if self.early_fusion:
            print('Using early fusion')
            self.scale_encoder = nn.Linear(scale_bins, self.posenc.embedding_dim).double()
            
            self.mini_transformer = MiniTransformer(input_dims=self.posenc.embedding_dim,
             forward_dims=self.posenc.embedding_dim, num_layers=self.num_t_layers, num_heads=8, num_tokens=2).double() # input shape = [batch, n_tokens, dim]
            self.fc_mu = nn.Linear(embed_dim, embed_dim).double()
            self.fc_logvar = nn.Linear(embed_dim, embed_dim).double()
        else:
            print('Using late fusion')
            self.scale_encoder = nn.Linear(scale_bins, embed_dim).double()
            self.fc_mu = nn.Linear(2*embed_dim, embed_dim).double()
            self.fc_logvar = nn.Linear(2*embed_dim, embed_dim).double()
        
        #define nn embeddings is scale encoding in learnable
        if self.scale_encoding=='learnable':
            self.learnable_scale_embeddings = nn.Embedding(3,scale_bins).double().to(self.device)


        #select the appropriate function deterministic vs probablistic that prepares
        #the embeddings for the appropriate loss
        self.loss_type=loss_type
        if loss_type=='probablistic':
            print('Using probablistic loss')
            self.loss_prep=self.probablistic_sapclip
        
        elif loss_type=='pcme':
            print('Using pcme loss')
            self.loss_prep=self.pcme_loss
            self.img_fc_mu = nn.Linear(embed_dim, embed_dim)
            self.img_fc_logvar = nn.Linear(embed_dim ,embed_dim)
            self.pcme_criterion = MCSoftContrastiveLoss()
            self.init_weights()
        
        elif loss_type=='pcme_uni':
            print('Using pcme uni loss')
            self.loss_prep= self.pcme_uni_loss
            self.img_fc_mu = nn.Linear(embed_dim, embed_dim)
            self.img_fc_logvar = nn.Linear(embed_dim ,embed_dim)
            self.pcme_criterion = MCSoftContrastiveLoss()
            self.init_weights()
        
        elif loss_type=='clip':
            print('Using CLIP loss')
            #initialize the logit scale
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 /0.07))
            del self.fc_mu
            del self.fc_logvar
        
        else:
            raise ValueError('Invalid Value for loss type')
        
       
    
    def init_weights(self):
        print('Initializing weights')
        r = np.sqrt(6.) / np.sqrt(self.fc_mu.in_features +
                                  self.fc_mu.out_features)
        # initialize the dist layers for location       
        self.fc_mu.weight.data.uniform_(-r, r)
        self.fc_mu.bias.data.fill_(-4)
        self.fc_logvar.weight.data.uniform_(-r, r)
        self.fc_logvar.bias.data.fill_(-4)
        #initialize the dist layers for image
        self.img_fc_mu.weight.data.uniform_(-r, r)
        self.img_fc_mu.bias.data.fill_(-4)
        self.img_fc_logvar.weight.data.uniform_(-r, r)
        self.img_fc_logvar.bias.data.fill_(-4)

    def dtype(self):
        if isinstance(self.visual, timm.models.vision_transformer.VisionTransformer):
            return self.visual.patch_embed.proj.weight.dtype
        elif self.vision_layers=='SatMAE':
            return self.visual.projection_layer.weight.dtype
        elif self.vision_layers=='CLIP':
            return self.visual.vision_model.visual_projection.weight.dtype
        else:
            return self.visual.conv1.weight.dtype


    def encode_image(self, image): 
        return self.visual(image.type(self.dtype()))

    def encode_location(self, coords, hot_scale):
        #check if the scale encoding is learnable
        if self.scale_encoding=='learnable':
            scale = self.learnable_scale_embeddings(hot_scale)
        else:
            scale = hot_scale
        
        #check if we are doing early fusion
        if self.early_fusion:
            scale_features = nn.functional.leaky_relu(self.scale_encoder(scale.double()))
            harmonics_features = self.posenc(coords.double())
            scale_harmonics_tokens = torch.stack([harmonics_features, scale_features], dim=1) 
            scaled_harmonics = self.mini_transformer(scale_harmonics_tokens.double())
            scaled_loc_features = self.nnet(scaled_harmonics)
        else:         
            location_features = nn.functional.leaky_relu(self.location(coords.double()))
            scale_features = nn.functional.leaky_relu(self.scale_encoder(scale.double()))
            scaled_loc_features = torch.cat([location_features, scale_features], dim=-1)
            # scaled_loc_features = location_features+scale_features
        
        if 'clip' in self.loss_type:
            return scaled_loc_features
        else:
            scaled_loc_mu = self.fc_mu(scaled_loc_features).double()
            scaled_loc_logvar = self.fc_logvar(scaled_loc_features).float()
            return [scaled_loc_mu, scaled_loc_logvar]
    
    def probablistic_sapclip(self, image_features, location_mu, location_logvar, label, intervals):
        #do not normalize the image features
        ##### to here ##########
        #get the dimension
        dim = location_mu.shape[-1]
        #compute standard deviation from log(variance)
        location_std = torch.exp(location_logvar/2)
        #get the distribution for computed mean and std
        location_dist = [torch.distributions.Normal(mu, std) for mu,std in zip(location_mu, location_std)]
        isotropic_dist = torch.distributions.Normal(torch.zeros(dim).to(self.device), torch.ones(dim).to(self.device))
        #compute KLD with isotropic normal
        # kld = torch.tensor([torch.distributions.kl.kl_divergence(loc_dist, isotropic_dist).sum() for loc_dist in location_dist]).mean()
        kld = compute_isotropic_kld(location_mu, location_std).mean()
        mask = label.unsqueeze(-1)
        #compute likelihood for each location
        likelihood_per_location = torch.zeros(len(intervals), len(intervals), device=self.device)
        for i, loc in enumerate(location_dist):
            # import code; code.interact(local=dict(globals(), **locals()))
            log_prob = loc.log_prob(image_features)
            summed_intervals = (log_prob*mask).sum(dim=1) #[batch_size, D]
            # Divide by the interval lengths to get the mean across the crops
            interval_lengths = intervals.view(-1, 1).float()
            mean_intervals = summed_intervals / interval_lengths #[batch_size, D]
            #finally sum across the dimensions per sample. We divide this by dims for numberical stability
            sum_log_prob = torch.sum(mean_intervals, dim=-1)/dim #[batch_size]
            #add log_prob to likelihood_per_location
            likelihood_per_location[i,:] = sum_log_prob
        return (likelihood_per_location, kld)
    
    def pcme_loss(self, image_features, location_mu, location_logsigma, intervals):
        batch_size, dim = location_mu.shape
        
        batch_size, dim = location_mu.shape
        img_mu = self.img_fc_mu(image_features)
        img_logsigma = self.img_fc_logvar(image_features)
        # generate samples from each distribution
        img_samples = sample_gaussian_tensors(img_mu, img_logsigma, 10) # shape = [total, 10, D]
        loc_samples = sample_gaussian_tensors(location_mu, location_logsigma, 10) # shape = [B, 10, D]

        index = torch.cumsum(intervals, dim=0)
        pruned_img_samples = torch.zeros_like(loc_samples)
        #loop over the img_samples and grab appropriate number of samples
        start = 0
        for j,i in enumerate(index):
            curr_tensor = (img_samples[start:i]).view(-1, dim)  #[scale, 10, D]
            permute_indices = torch.randperm(curr_tensor.shape[0])[:10]
            curr_tensor = curr_tensor[permute_indices]
            pruned_img_samples[j] = curr_tensor

        pcme_loss, pcme_loss_dict = self.pcme_criterion(pruned_img_samples, loc_samples, img_logsigma, location_logsigma, img_mu, location_mu)

        return (pcme_loss, pcme_loss_dict)
    
    #loss when only single image is sampled for each loc irrespective scale
    def pcme_uni_loss(self, image_features, location_mu, location_logsigma, intervals):
        batch_size, dim = location_mu.shape
        img_mu = self.img_fc_mu(image_features)
        img_logsigma = self.img_fc_logvar(image_features)
        # generate samples from each distribution
        img_samples = sample_gaussian_tensors(img_mu, img_logsigma, 8) # shape = [B, 10, D]
        loc_samples = sample_gaussian_tensors(location_mu, location_logsigma, 8) # shape = [B, 10, D]

        pcme_loss, pcme_loss_dict = self.pcme_criterion(img_samples, loc_samples, img_logsigma, location_logsigma, None, None)
        return (pcme_loss, pcme_loss_dict)
    
    def clip_loss(self, img_to_loc_similarity):
        img_to_loc_loss = torch.nn.functional.cross_entropy(img_to_loc_similarity, torch.arange(len(img_to_loc_similarity)).to(self.device))
        loc_to_img_loss = torch.nn.functional.cross_entropy(img_to_loc_similarity.t(), torch.arange(len(img_to_loc_similarity)).to(self.device))
        clip_loss = (img_to_loc_loss + loc_to_img_loss)/2.0
        return clip_loss

    def forward(self, batch):
        image = batch['image']
        coords = batch['point']
        
        hot_scale = batch['hot_scale']
        scale = batch['scale']
        label = batch['label']

        #compute embeddings from both directions
        image_features = self.encode_image(image)
        if self.loss_type=='clip':
            loc_features = self.encode_location(coords,hot_scale)
            #compute cosine similarities
            image_features = image_features/image_features.norm(p=2, dim=-1, keepdim=True)
            loc_features = loc_features/loc_features.norm(p=2, dim=-1, keepdim=True)
            img_to_loc_sim = image_features.double() @ loc_features.t()
            #get the logit scale
            logit_scale = self.logit_scale.exp()
            img_to_loc_sim = img_to_loc_sim*logit_scale
            #compute clip loss
            clip_loss = self.clip_loss(img_to_loc_sim)
            clip_dict = {'logit_scale':1/self.logit_scale.data.exp()}
            return (clip_loss, clip_dict)
        else:
            mu, logvar = self.encode_location(coords, hot_scale)
            if self.loss_type=='pcme' or self.loss_type=='pcme_uni':
                pcme_loss, pcme_loss_dict = self.loss_prep(image_features, mu, logvar, scale)
                return (pcme_loss, pcme_loss_dict)
            else:
                #compute likelihood per location for each sample [batch_size, batch_size]
                likelihood_per_location, kld_loss = self.loss_prep(image_features, mu, logvar, scale)
                logit_scale = self.logit_scale.exp()

                likelihood_per_location = likelihood_per_location*logit_scale

                #compute contrastive loss
                contrastive_loss = torch.nn.functional.cross_entropy(likelihood_per_location, torch.eye(likelihood_per_location.shape[0], device=self.device))
                
                return contrastive_loss, kld_loss


def deterministic_sapclip(image_features, location_features, label):
    # normalized features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    location_features = location_features / location_features.norm(dim=1, keepdim=True)
    # this will be of shape [B,B'], where B' > B
    location_to_image_similarity = location_features@image_features.t()
    image_to_location_similarity = location_to_image_similarity.t() # [B',B]

    #normalize the label to distribute the probablity between the positive images
    location_to_image_label = label/torch.sum(label, dim=1, keepdim=True)
    image_to_location_label = label.t() 
    return location_to_image_similarity, location_to_image_label, image_to_location_similarity, image_to_location_label  # label shape [B,B']
    
    
def convert_weights(model: nn.Module):


    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

if __name__ == '__main__':
    # data_root = '/scratch/a.dhakal/hyper_satclip/data/satclip_data/satclip_sentinel/images'
    data_root = '/projects/bdec/adhakal2/hyper_satclip/data/satclip_sentinel/images'
    device = 'cpu'
    embed_dim=256
    image_resolution=256
    vision_layers='CLIP'
    vision_width=768
    vision_patch_size=32
    in_channels=4
    le_type="grid"
    pe_type="siren"
    frequency_num=16
    max_radius=260
    min_radius=1
    legendre_polys=16
    harmonics_calculation="analytic"
    sh_embedding_dims=32
    learning_rate=1e-4
    weight_decay=0.01
    num_hidden_layers=2
    capacity=256
    model = SatCLIP_2(
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
            loss_type='clip',
            satclip_pretrained=True,
            early_fusion=True,    
            num_t_layers=1,
            device=device,

        )

    dataset = SAPCLIP_Dataset(root=data_root, transform_type='sapclip_uni', crop_size=224, prototype=False,scale_bins=50)
    train_loader, val_loader = get_split_dataset(dataset,transform_type='sapclip_uni', val_split=0.05, batch_size=8,
     num_workers=0)
    
    batch = next(iter(train_loader))
    # visual_model = model.visual

    output = model(batch)
    import code; code.interact(local=dict(globals(), **locals()))
    
    
    



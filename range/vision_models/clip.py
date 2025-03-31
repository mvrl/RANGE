import numpy as np
import torch.nn as nn
from transformers import CLIPVisionModelWithProjection
import pytorch_lightning as pl
import code

class Clip(pl.LightningModule):
    def __init__(self, embed_dim=256, device='cuda', vit_type='16'):
        super().__init__()
        
        self.vit_map = {'32':'openai/clip-vit-base-patch32', '16':'openai/clip-vit-base-patch16', '14L':'openai/clip-vit-large-patch14'}
        # print(f'Args.vit is {args.vit}')
        # self.vit = self.vit_map[args.vit]
        self.vision_model = CLIPVisionModelWithProjection.from_pretrained(self.vit_map[vit_type]).train()
        self.projection_layer = nn.Linear(512, embed_dim)
    def forward(self,x):
        batch_tensors = self.vision_model(x)
        output = batch_tensors.image_embeds
        output = self.projection_layer(output)
        return output

if __name__ == '__main__':
    model = Clip(256, 'cuda', '16')
    import code; code.interact(local=dict(globals(), **locals()))
    

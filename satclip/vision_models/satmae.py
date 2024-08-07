from rshf.satmae import SatMAE as Satmae
import torch.nn as nn
import torch

class SatMAE(nn.Module):
    def __init__(self, embed_dim=256, device='cuda'):
        super().__init__()
        
        self.vision_model = Satmae.from_pretrained("MVRL/satmae-vitlarge-fmow-pretrain-800")
        self.projection_layer = nn.Linear(1024, embed_dim)
    def forward(self,x):
        batch_tensors = self.vision_model.forward_encoder(x, mask_ratio=0.0)[0]
        import code; code.interact(local=dict(globals(), **locals()))
        pooled_tensors = batch_tensors[:,1:,:].mean(dim=1)
        output = self.projection_layer(pooled_tensors)
        return output

if __name__ == '__main__':
    model = SatMAE(256, 'cuda', '16')
    data = torch.rand(2,3,224,224)
    model(data)
    
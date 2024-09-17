#local imports
from ..fit import SAPCLIP_PCME
import torch

def load_checkpoint(ckpt_path, device='cuda'):
    model = SAPCLIP_PCME.load_from_checkpoint(ckpt_path,
    map_location=torch.device(device))
    return model.eval()

if __name__ == '__main__':
    ckpt_path = '/projects/bdec/adhakal2/hyper_satclip/logs/SAPCLIP/0tflzztx/checkpoints/epoch=162-acc_eco=0.000.ckpt'
    import code; code.interact(local=dict(globals(), **locals()))
    model = load_checkpoint(ckpt_path, 'cpu')
    from einops import repeat
    inp_coord = torch.rand(64,2).to('cuda')
    inp_scale = torch.tensor([0,0,1])
    inp_scale = repeat(inp_scale, 'd -> b d', b=64).cuda()
    out = model.encode_location(inp_coord, inp_scale)

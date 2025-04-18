#local imports
from ..fit import SAPCLIP_PCME
import torch

def load_checkpoint(ckpt_path, device='cuda'):
    ckpt = torch.load(ckpt_path, map_location=torch.device(device))
    hyper_parameters = ckpt['hyper_parameters']
    hyper_parameters['device'] = device

    model = SAPCLIP_PCME(**hyper_parameters)
    model.load_state_dict(ckpt['state_dict'])
    return model.eval()

if __name__ == '__main__':
    device='cpu'
    ckpt_path = '/projects/bdec/adhakal2/hyper_satclip/logs/SAPCLIP/0tflzztx/checkpoints/epoch=162-acc_eco=0.000.ckpt'
    model = load_checkpoint(ckpt_path, 'cpu')
    import code; code.interact(local=dict(globals(), **locals()))
    from einops import repeat
    inp_coord = torch.rand(64,2).to('cuda')
    inp_scale = torch.tensor([0,0,1])
    inp_scale = repeat(inp_scale, 'd -> b d', b=64).cuda()
    out = model.encode_location(inp_coord, inp_scale)


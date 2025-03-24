import torch.nn as nn
import torch
import numpy as np

class temp_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 /0.07))
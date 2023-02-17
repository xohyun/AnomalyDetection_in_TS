import torch.nn as nn
import torch

class Weight_blocks(nn.Module):
    def __init__(self, recon, forecast, var):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1,1,1))
    def forward():
        pass
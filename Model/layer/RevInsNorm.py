'''
2023.02.15
- Reverse Instance Normalize layer
'''

import torch
import torch.nn as nn


class RIN(nn.Module):
    def __init__(self):
        super(RIN, self).__init__()
        self.affine_weight = nn.Parameter(torch.ones(1, 1, 1))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, 1))

    def forward(self, x, flag):
        means = x.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)

        if flag == 'in':
            x = x - means
            x /= stdev
            x = x * self.affine_weight + self.affine_bias
            return x, means, stdev

        elif flag == 'out':
            x = x - self.affine_bias
            x = x / (self.affine_weight + 1e-10)
            x = x * stdev
            x = x + means
            return x, means, stdev

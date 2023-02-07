import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

class PositionalEncoding(nn.Module):
    
    def __init__(self, seq_len, d_model, n):
        
        super(PositionalEncoding, self).__init__() # nn.Module 초기화
        
        # encoding : (seq_len, d_model)
        self.encoding = torch.zeros(seq_len, d_model)
        self.encoding.requires_grad = False
        
        # (seq_len, )
        pos = torch.arange(0, seq_len)
        # (seq_len, 1)         
        pos = pos.float().unsqueeze(dim=1) # int64 -> float32 (없어도 되긴 함)
        
        _2i = torch.arange(0, d_model, step=2).float()
        
        self.encoding[:, ::2] = torch.sin(pos / (n ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (n ** (_2i / d_model)))
        
        
    def forward(self, x):
        # x.shape : (batch, seq_len) or (batch, seq_len, d_model)
        seq_len = x.size()[1] 
        # return : (seq_len, d_model)
        # return matrix will be added to x by broadcasting
        return self.encoding[:seq_len, :]

if __name__ == "__main__":
    seq_len = 100
    d_model = 512
    sample_pos_encoding = PositionalEncoding(seq_len=seq_len, d_model=d_model, n=10000)
    x = torch.rand(10, seq_len, d_model)
    print(x.shape)
    # torch.Size([10, 100, 512])
    print("?>>>", sample_pos_encoding(x).shape)

    x_added_PE = x + sample_pos_encoding(x)
    print(x_added_PE.shape)
    # torch.Size([10, 100, 512])
import torch
import torch.nn as nn

# https://github.com/ServiceNow/N-BEATS/blob/master/models/nbeats.py

class ModelBlock(nn.Module):
    def __init__(self, input_size, theta_size: int, layers: int, layer_size: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(in_features=input_size, out_features=layer_size)] +
                                      [torch.nn.Linear(in_features=layer_size, out_features=layer_size)
                                       for _ in range(layers - 1)])
        self.basis_parameters = torch.nn.Linear(in_features=layer_size, out_features=theta_size)
    
    def forward(self, x):
        block_input = x
        for layer in self.layers:
            block_input = torch.relu(layer(block_input))
        basis_parameters = self.basis_parameters(block_input)
        return self.basis_function(basis_parameters)
##    MLP    ##

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(256, 256)):
        super(MLP, self).__init__()
        
        layers = []
        # dims = [input_dim] + list(hidden_dims)
        last_dim = input_dim
        
        # for i in range(len(dims) - 1):
            # layers.append(nn.Linear(dims[i], dims[i+1]))
            # layers.append(nn.ReLU(implace=True))
        
        for i in hidden_dims:
            layers.append(nn.Linear(last_dim, i))
            layers.append(nn.ReLU())
            last_dim = i
            
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)
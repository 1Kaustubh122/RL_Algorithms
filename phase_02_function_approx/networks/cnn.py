##    CNN    ##

import torch
import numpy as np
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_channels,output_dim):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64,64, kernel_size=3),
            nn.ReLU()
        )
        
        self.fc_input_dim = self._get_conv_output_dim(input_channels)
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        
    def _get_conv_output_dim(self, input_channels):
        dummy_input = torch.zeros(1, input_channels, 84, 84)
        out = self.conv_layers(dummy_input)
        return int(torch.flatten(out, start_dim=1).shape[1])

    def forward(self, x):
        conv_out = self.conv_layers(x)
        flat = torch.flatten(conv_out, start_dim=1)
        return self.fc(flat)
    
class QNetwork(nn.Module):
    def __init__(self, obs_shape, num_actions):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, state):
        return self.model(state)

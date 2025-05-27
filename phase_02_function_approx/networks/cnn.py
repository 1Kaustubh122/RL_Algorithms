##    CNN    ##

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(self, output_dim):
        super(CNNEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64, kernel_size=3),
            nn.ReLU(inplace=True)
        )
        
        self.fc = nn.Linear(64 * 7 * 7, output_dim)
        
    def forward(self, x):
        x = x/255.0
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
class CNNQNetwork(nn.Module):
    def __init__(self, action_dim, feature_dim=512):
        super(CNNQNetwork, self).__init__()
        self.encoder = CNNEncoder(output_dim=feature_dim)
        self.q_head = nn.Linear(feature_dim, action_dim)

    def forward(self, obs):
        features = self.encoder(obs)
        q_values = self.q_head(features)
        return q_values
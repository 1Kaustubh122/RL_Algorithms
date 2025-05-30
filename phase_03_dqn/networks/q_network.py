import torch.nn as nn
    
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
        self._init_weights()
        
    def _init_weights(self):
        for layers in self.model:
            if isinstance(layers, nn.Linear):
                nn.init.kaiming_uniform_(layers.weight)
                nn.init.zeros_(layers.bias)

    def forward(self, state):
        return self.model(state)
    
                
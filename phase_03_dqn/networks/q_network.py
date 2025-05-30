import torch.nn as nn
    
class QNetwork(nn.Module):
    def __init__(self, obs_shape, num_actions):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        self._init_weights()
        
    def _init_weights(self):
        for layers in self.model:
            if isinstance(layers, nn.Linear):
                nn.init.kaiming_uniform_(layers.weight)
                nn.init.zeros_(layers.bias)

    def forward(self, state):
        return self.model(state)
    


class DuelingQNetwork(nn.Module):
    def __init__(self, obs_shape, num_actions):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_shape, 128),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )


    def forward(self, state):
        features = self.feature(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

            
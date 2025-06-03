import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, is_continous: bool):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.is_continous = is_continous
        
        self.shared = nn.Sequential(
            nn.Linear(in_features=obs_dim, out_features=64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        
        if is_continous:
            self.mean_layer = nn.Linear(64, action_dim)
            self.log_std_layer = nn.Linear(64, action_dim)
        else:
            self.logits_layer = nn.Linear(64, action_dim)
        
    def get_action(self, state: torch.tensor):
        dist = self.forward(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        if self.is_continous:
            log_prob = log_prob.sum(dim=-1)
            entropy = entropy.sum(dim=-1)
            
        return action, log_prob, dist, entropy
    
    def forward(self, state):
        x = self.shared(state)
        if self.is_continous:
            mean = self.mean_layer(x)
            log_std = torch.clamp(self.log_std_layer(x), min=-20, max=2)
            std = torch.exp(log_std)
            return torch.distributions.Normal(mean, std)
        else:
            logits = self.logits_layer(x)
            return torch.distributions.Categorical(logits=logits)


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.model(x).squeeze(-1)
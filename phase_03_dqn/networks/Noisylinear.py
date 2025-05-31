import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    def __init__(self, in_feat, out_feat, sigma_init=0.5):
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.empty(out_feat, in_feat))
        self.weight_sigma = nn.Parameter(torch.empty(out_feat, in_feat))
        self.register_buffer("weight_epsilon", torch.empty(out_feat, in_feat))

        self.bias_mu = nn.Parameter(torch.empty(out_feat))
        self.bias_sigma = nn.Parameter(torch.empty(out_feat))
        self.register_buffer("bias_epsilon", torch.empty(out_feat))

        self.reset_parameters()
        self.reset_noise()
        
    def reset_parameters(self):
        mu_range = 1 / self.in_feat ** 0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.sigma_init / self.in_feat ** 0.5)
        self.bias_sigma.data.fill_(self.sigma_init / self.in_feat ** 0.5)

        
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()
        
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_feat)
        epsilon_out = self._scale_noise(self.out_feat)

        self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

        
    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            w = self.weight_mu
            b = self.bias_mu

        return F.linear(x, w, b)
#!/usr/bin/env python3
import torch
from torch import nn
import casadi as ca

# Sinusoidal activation
class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class mlp(nn.Module):
    def __init__(self, input_layer_size=8, output_layer_size=1, hidden_layer_sizes=[100, 100, 100], activation='sin', batch_norm=False):
        super(mlp, self).__init__()
        layers = [nn.Linear(input_layer_size, hidden_layer_sizes[0], dtype=torch.double)]
        
        # Possibly add batch normalization
        if batch_norm:
            layers.append(nn.BatchNorm1d(num_features=hidden_layer_sizes[0], dtype=torch.double))
        
        # Apply activation function
        if activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'relu':
            layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))
        elif activation == 'sin':
            layers.append(Sin())
        
        for i in range(len(hidden_layer_sizes)-1):
            layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1], dtype=torch.double))
            
            # Possibly add batch normalization
            if batch_norm:
                layers.append(nn.BatchNorm1d(num_features=hidden_layer_sizes[0], dtype=torch.double))
            
            # Apply activation function
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))
            elif activation == 'sin':
                layers.append(Sin())
        
        layers.append(nn.Linear(hidden_layer_sizes[-1], output_layer_size, dtype=torch.double))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

    def get_casadi_mlp(self):
        modules = list(self.modules())[2:]
        sym_in = ca.SX.sym('sym_in', modules[0].in_features)
        x = sym_in
        for m in modules:
            if m._get_name() == 'Linear':
                W = m.weight.cpu().detach().numpy()
                b = m.bias.cpu().detach().numpy()
                x = W @ x + b
            if m._get_name() == 'Sin':
                x = ca.sin(x)
            if m._get_name() == 'Tanh':
                x = ca.tanh(x)

        return ca.Function('mlp_ca', [sym_in], [x])
    
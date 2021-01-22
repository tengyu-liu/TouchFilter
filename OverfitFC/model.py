import torch
from torch import nn
import numpy as np


class OverfitSDF(nn.Module):
    def __init__(self, n_obj):
        super(OverfitSDF, self).__init__()
        self.weights = nn.ParameterList([
            nn.Parameter(torch.Tensor(n_obj, 3, 32)), 
            nn.Parameter(torch.Tensor(n_obj, 32, 32)), 
            nn.Parameter(torch.Tensor(n_obj, 32, 32)), 
            nn.Parameter(torch.Tensor(n_obj, 32, 32)), 
            nn.Parameter(torch.Tensor(n_obj, 32, 32)), 
            nn.Parameter(torch.Tensor(n_obj, 32, 32)), 
            nn.Parameter(torch.Tensor(n_obj, 32, 32)), 
            nn.Parameter(torch.Tensor(n_obj, 32, 32)), 
            nn.Parameter(torch.Tensor(n_obj, 32, 1))
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.Tensor(n_obj, 1, 32)),
            nn.Parameter(torch.Tensor(n_obj, 1, 32)),
            nn.Parameter(torch.Tensor(n_obj, 1, 32)),
            nn.Parameter(torch.Tensor(n_obj, 1, 32)),
            nn.Parameter(torch.Tensor(n_obj, 1, 32)),
            nn.Parameter(torch.Tensor(n_obj, 1, 32)),
            nn.Parameter(torch.Tensor(n_obj, 1, 32)),
            nn.Parameter(torch.Tensor(n_obj, 1, 32)),
            nn.Parameter(torch.Tensor(n_obj, 1, 1))
        ])
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        for w in self.weights:
            nn.init.xavier_normal_(w)
        for b in self.biases:
            nn.init.constant_(b, 0)
    
    def forward(self, pts, idx):
        # pts: B x N x 3
        # idx: B
        x = self.relu(torch.einsum('ijk,ikm->ijm', pts, self.weights[0][idx]) + self.biases[0][idx]) # in: B x N x 3 , B x 3 x 32 -> out: B x N x 32
        x = self.relu(torch.einsum('ijk,ikm->ijm', x, self.weights[1][idx]) + self.biases[1][idx]) 
        x = self.relu(torch.einsum('ijk,ikm->ijm', x, self.weights[2][idx]) + self.biases[2][idx]) 
        x = self.relu(torch.einsum('ijk,ikm->ijm', x, self.weights[3][idx]) + self.biases[3][idx]) 
        x = self.relu(torch.einsum('ijk,ikm->ijm', x, self.weights[4][idx]) + self.biases[4][idx]) 
        x = self.relu(torch.einsum('ijk,ikm->ijm', x, self.weights[5][idx]) + self.biases[5][idx]) 
        x = self.relu(torch.einsum('ijk,ikm->ijm', x, self.weights[6][idx]) + self.biases[6][idx]) 
        x = self.relu(torch.einsum('ijk,ikm->ijm', x, self.weights[7][idx]) + self.biases[7][idx]) 
        x = torch.einsum('ijk,ikm->ijm', x, self.weights[8][idx]) + self.biases[8][idx]
        return x.squeeze(-1)
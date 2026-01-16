
import torch
import torch.nn as nn

class Perceptron(nn.Module):
    """
    Perceptron (P) - Basic single-layer feedforward network.
    The simplest neural network with configurable activation functions.
    """
    def __init__(self, input_dim, output_dim, activation='sigmoid', output_type='prediction'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_type = output_type
        
        self.linear = nn.Linear(input_dim, output_dim)
        
        self.act_fn = {
            'sigmoid': torch.sigmoid,
            'relu': torch.relu,
            'tanh': torch.tanh,
            'step': lambda x: (x > 0).float()
        }.get(activation.lower(), torch.sigmoid)
        
    def forward(self, x, return_embedding=None, return_sequence=False):
        # Handle sequence
        if x.dim() == 3:
            b, s, f = x.shape
            x_flat = x.view(b * s, f)
            out = self.act_fn(self.linear(x_flat))
            res = out.view(b, s, -1)
        else:
            res = self.act_fn(self.linear(x))
            
        if not return_sequence and res.dim() == 3:
            return res[:, -1, :]
        return res

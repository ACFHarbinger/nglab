"""
LSTM Model for Time Series.
"""
import torch
import torch.nn as nn

from torch.autograd import Variable


class LSTM(nn.Module):
    """
    Long Short-Term Memory (LSTM) network for sequence processing.
    """
    def __init__(self, input_dim, hidden_dim, embed_dim, n_layers, output_dim, n_heads=8):
        """
        Initialize the LSTM.

        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden state dimension.
            embed_dim (int): Embedding dimension (unused, for compatibility).
            n_layers (int): Number of LSTM layers.
            output_dim (int): Output dimension.
            n_heads (int): Number of attention heads (unused, for compatibility).
        """
        super(LSTM,self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input sequence [batch, seq_len, input_dim].

        Returns:
            Tensor: Output [batch, output_dim].
        """
        h0 = Variable(torch.zeros(self.n_layers, x.size(0), self.hidden_dim)).to(x.device)
        c0 = Variable(torch.zeros(self.n_layers, x.size(0), self.hidden_dim)).to(x.device)
        out, (h_out, c_out) = self.lstm(x,(h0,c0))
        out = self.fc1(out[:,-1,:])
        return out.squeeze(1)
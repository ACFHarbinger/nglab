#!/usr/bin/env python

from torch import nn

class RNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout_rate=0.2):
        super(RNNLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.feature_extractor = nn.Sequential(
            nn.LSTMCell(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.LSTMCell(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.latent_fc = nn.Linear(hidden_dim, latent_dim)

    def set_latent_dim(self, latent_dim):
        self.latent_fc = nn.Linear(self.hidden_dim, latent_dim)

    def forward(self, x):
        h = self.feature_extractor(x)
        return self.latent_fc(h)
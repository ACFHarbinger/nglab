import torch
import torch.nn as nn

from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim):
        super(LSTM,self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)

        self.fc1 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self,x):
        h0 = Variable(torch.zeros(self.layers, x.size(0), self.hidden_dim)).to(self.device)

        c0 = Variable(torch.zeros(self.layers, x.size(0), self.hidden_dim)).to(self.device)
        out, (h_out, c_out) = self.lstm(x,(h0,c0))
        out = self.fc1(out[:,-1,:])
        return out.squeeze(1)
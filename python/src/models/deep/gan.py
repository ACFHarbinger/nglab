"""
Time Series GAN Networks (Generator and Discriminator).
"""
import torch
import torch.nn as nn


class TimeGANGenerator(nn.Module):
    """
    Generator for Time Series Forecasting GAN.
    Seq2Seq LSTM Architecture: Encoder -> Decoder.
    Takes history X_{1:t} and generates X_{t+1:t+k}.
    """
    def __init__(self, input_dim, output_dim, seq_len, pred_len, hidden_dim=64, n_layers=2):
        """
        Args:
            input_dim (int): Feature dimension of input.
            output_dim (int): Feature dimension of output.
            seq_len (int): Input sequence length (history).
            pred_len (int): Output sequence length (horizon).
            hidden_dim (int): LSTM hidden dimension.
            n_layers (int): Number of LSTM layers.
        """
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        
        # Decoder
        # If we use an autoregressive decoder, we need more logic.
        # Simple approach: Project encoded state to initial state of decoder?
        # Or just use the final hidden state of encoder as initial state for decoder.
        # We need an input to the decoder. 
        # Strategy: Use zero inputs or last observation repeated?
        
        self.decoder = nn.LSTM(output_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """
        x: (Batch, Seq_Len, Features)
        Returns: (Batch, Pred_Len, Features)
        """
        batch_size = x.size(0)
        
        # Encode
        _, (h_n, c_n) = self.encoder(x)
        
        # Decode
        # We need to generate 'pred_len' steps.
        # Input to decoder:
        # Option A: Zeros.
        # Option B: Last value of x repeated (if standard forecasting).
        # Let's use last value of x as first input, then autoregressive? 
        # Ideally simple GAN generator outputs the whole sequence at once or autoregressively.
        # For 'TimeGAN' style, usually the generator is an RNN that outputs the sequence.
        
        # Simplified vector output approach: Use last hidden state to predict all steps?
        # No, we want temporal structure.
        
        # Let's seed decoder with last observation
        curr_input = x[:, -1:, :] # (B, 1, F)
        
        outputs = []
        
        # State
        h_state, c_state = h_n, c_n
        
        for _ in range(self.pred_len):
            out, (h_state, c_state) = self.decoder(curr_input, (h_state, c_state))
            # out: (B, 1, Hidden)
            pred = self.fc(out) # (B, 1, Out_F)
            outputs.append(pred)
            curr_input = pred # Autoregressive interaction
            
        outputs = torch.cat(outputs, dim=1) # (B, Pred_Len, F)
        return outputs

class TimeGANDiscriminator(nn.Module):
    """
    Discriminator for Time Series GAN.
    Takes full sequence (History + Future) and classifies as Real/Fake.
    Architecture: Bidirectional LSTM.
    """
    def __init__(self, input_dim, hidden_dim=64, n_layers=2):
        """
        Args:
            input_dim (int): Feature dimension.
        """
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1) # *2 for bidirectional
        
    def forward(self, x):
        """
        x: (Batch, Seq_Len + Pred_Len, Features)
        Returns: (Batch, 1) - Logits
        """
        # LSTM
        # output: (B, L, 2*H), (not needed)
        # We can pool or take last state.
        # Bi-LSTM: Last state contains forward of last step and backward of first step.
        # Easier to max pool over time or take mean.
        
        output, _ = self.lstm(x) # (B, L, 2*H)
        
        # Global Average Pooling over time to catch anomalies anywhere
        # Or just use the last time step?
        # Pooling is robust.
        # out_pooled = torch.mean(output, dim=1) 
        
        # Often in TimeGAN implementation, they return sequence of scores or score for entire sequence.
        # Let's return score for entire sequence.
        
        # Taking the last time step of forward and first of backward is standard for classification,
        # but mean pooling works well for "is this sequence realistic?".
        
        out_pooled = torch.mean(output, dim=1)
        logits = self.fc(out_pooled)
        return logits

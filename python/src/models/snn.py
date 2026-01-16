
"""
Spiking Neural Network (SNN) implementation using Surrogate Gradients.
"""
import torch
import torch.nn as nn
import math

class SurrogateHeaviside(torch.autograd.Function):
    """
    Heaviside step function with a surrogate gradient for backpropagation.
    We use the Fast Sigmoid derivative as the surrogate gradient.
    """
    @staticmethod
    def forward(ctx, input, alpha=25.0):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Surrogate gradient: alpha / (1 + |alpha * input|)^2
        # This is the derivative of the fast sigmoid function.
        grad_input = grad_output * (ctx.alpha / (1 + torch.abs(ctx.alpha * input)).pow(2))
        return grad_input, None

def surrogate_heaviside(x, alpha=25.0):
    return SurrogateHeaviside.apply(x, alpha)


class LIFCell(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) Neuron Cell.
    
    Dynamics:
    U[t] = decay * U[t-1] + W X[t] - S[t-1] * threshold
    S[t] = Heaviside(U[t] - threshold)
    """
    def __init__(self, input_dim, hidden_dim, decay=0.9, threshold=1.0, alpha=25.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.decay = decay
        self.threshold = threshold
        self.alpha = alpha
        
        self.linear = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x, state=None):
        """
        Args:
            x: Input tensor (Batch, Input_Dim)
            state: Tuple (u, s) of previous membrane potential and spike.
                   If None, initialized to zeros.
        Returns:
            s_next: Spikes at current time step (Batch, Hidden_Dim)
            state_next: (u_next, s_next)
        """
        batch_size = x.size(0)
        
        if state is None:
            u = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            s = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            # We don't necessarily need previous s for reset if we implement soft reset:
            # U[t] = decay * U[t-1] + input - s[t-1] * threshold
            # But standard soft reset usually resets AFTER spike generation of current step.
            # Let's use: U[t] = decay * (U[t-1] - S[t-1]*threshold) + Input[t]
            # This is "reset by subtraction".
        else:
            u, s = state
            
        # Linear transform of input
        i_inj = self.linear(x)
        
        # Update membrane potential
        # Reset mechanism: subtract threshold if previous step invoked a spike
        u_next = self.decay * (u - s * self.threshold) + i_inj
        
        # Fire spike
        s_next = surrogate_heaviside(u_next - self.threshold, self.alpha)
        
        return s_next, (u_next, s_next)


class SNN(nn.Module):
    """
    Multi-layer Spiking Neural Network.
    Currently implements a single recurrent layer of LIF cells.
    """
    def __init__(self, input_dim, hidden_dim, n_layers=1, output_dim=None, 
                 decay=0.9, threshold=1.0, alpha=25.0, dropout=0.0, output_type='prediction'):
        """
        Args:
            input_dim: Feature dimension
            hidden_dim: Hidden dimension of LIF neurons
            n_layers: Number of LIF layers (stacked) -- simplified to 1 for now or stacked?
                      Let's implement Single Layer Recurrent SNN first.
            output_dim: Output dimension (if prediction)
            decay: Membrane potential decay factor
            threshold: Firing threshold
            alpha: Surrogate gradient steepness
            output_type: 'prediction' or 'embedding'
        """
        super().__init__()
        self.output_type = output_type
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers # currently handling 1 recurrent layer logic primarily
        
        # We can implement stacked LIF cells
        self.lif_layers = nn.ModuleList([
            LIFCell(
                input_dim if i == 0 else hidden_dim,
                hidden_dim,
                decay=decay,
                threshold=threshold,
                alpha=alpha
            ) for i in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Optional readout layer (leaky calculation or simple linear?)
        # For prediction, we usually take rate or membrane potentials.
        # Here we follow existing backbone pattern: Linear projection of last state (spikes or potential).
        # Using potential (u) often carries more information for regression than binary spikes.
        # But if 'embedding' is spikes, it's very sparse.
        self.fc = nn.Linear(hidden_dim, output_dim) if output_dim else None
        
    def forward(self, x, return_embedding=None, return_sequence=False):
        """
        Args:
            x: (Batch, Seq, Feature)
            
        Returns:
            Output compatible with TimeSeriesBackbone
        """
        batch_size, seq_len, _ = x.size()
        
        # Initialize states for each layer
        states = [None] * self.n_layers
        
        # We need to collect outputs to return sequence
        # If n_layers > 1, we feed output of layer l-1 to layer l
        
        # Let's simple loop over time, then layers (standard RNN processing)
        
        # Or loop over layers, then time? (Easier for stacking if no feedback across layers)
        # Yes, standard stacked RNN.
        
        current_input = x
        
        all_layer_outputs = [] # For last layer
        
        for layer_idx, lif in enumerate(self.lif_layers):
            layer_outputs = []
            state = None
            for t in range(seq_len):
                input_t = current_input[:, t, :]
                spikes, state = lif(input_t, state)
                layer_outputs.append(spikes)
            
            # Stack spikes for next layer input: (Batch, Seq, Hidden)
            current_input = torch.stack(layer_outputs, dim=1)
            
            if layer_idx < self.n_layers - 1:
                current_input = self.dropout(current_input)
                
        # Final layer output is current_input (spikes)
        # We might also want membrane potentials for smoother regression? 
        # For now, let's stick to spikes as the "SNN" output.
        
        final_spikes = current_input
        
        if return_sequence:
            out = final_spikes
        else:
            out = final_spikes[:, -1, :]
            
        should_return_embedding = return_embedding if return_embedding is not None else (self.output_type == 'embedding')

        if should_return_embedding:
            return out
            
        # If prediction, project
        if self.fc is not None:
             return self.fc(out)
        else:
             # Fallback if no output_dim given but prediction requested?
             return out

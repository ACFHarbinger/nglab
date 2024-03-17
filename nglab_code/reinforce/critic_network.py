from torch import nn
from ..models.graph_encoder import GraphAttentionEncoder


# Attention, Learn to Solve Routing Problems and Heterogeneous Attentions for Solving PDP via DRL
class CriticNetwork(nn.Module):

    def __init__(
        self,
        method,
        input_dim,
        embedding_dim,
        hidden_dim,
        n_layers,
        encoder_normalization
    ):
        super(CriticNetwork, self).__init__()

        self.hidden_dim = hidden_dim

        self.encoder = GraphAttentionEncoder(
            method=method,
            node_dim=input_dim,
            n_heads=8,
            embed_dim=embedding_dim,
            n_layers=n_layers,
            normalization=encoder_normalization
        )

        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs):
        """

        :param inputs: (batch_size, graph_size, input_dim)
        :return:
        """
        _, graph_embeddings = self.encoder(inputs)
        return self.value_head(graph_embeddings)

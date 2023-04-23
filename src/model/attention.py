import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, input_size):
        """Single attention layer (not mulithead attention).

        Args:
            input_size: dimensionality of input vectors
            context_length: number of input vectors to consider in a given sequence
        """
        super().__init__()
        self.W1 = nn.Linear(in_features=input_size, out_features=input_size)
        self.w2 = nn.Linear(in_features=input_size, out_features=1)

        self.softmax = nn.Softmax(dim=1)
        self.dk = torch.sqrt(torch.tensor(input_size, dtype=torch.float32))

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: torch.Tensor (*,TEMPORAL_DIM, input_size)
        """
        weights = self.W1(hidden_states)
        weights = torch.tanh(weights)
        weights = self.w2(weights) / self.dk
        weights = self.softmax(weights)
        out = torch.sum(weights * hidden_states, dim=1)
        return out

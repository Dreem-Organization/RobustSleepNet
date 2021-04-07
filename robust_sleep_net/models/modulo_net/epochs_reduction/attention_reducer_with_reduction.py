from torch import nn

from ..modules import Attention


class AttentionReducer(nn.Module):
    def __init__(self, group, output_dim, context_size=32, activation="tanh"):
        super(AttentionReducer, self).__init__()
        channels = group["reducer_input_shape"][0]
        self.reduction = nn.Linear(channels, output_dim)
        self.attention = Attention(output_dim, context_size, activation=activation)
        self.output_dim = output_dim

    def forward(self, x):
        x = self.reduction(x.transpose(1, 2))
        x, _ = self.attention(x)
        return x

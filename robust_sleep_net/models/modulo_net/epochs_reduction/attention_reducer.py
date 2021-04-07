from torch import nn

from ..modules import Attention


class AttentionReducer(nn.Module):
    def __init__(self, group, context_size=32, activation="tanh"):
        super(AttentionReducer, self).__init__()
        channels = group["reducer_input_shape"][0]
        self.attention = Attention(channels, context_size, activation=activation)
        self.output_dim = channels

    def forward(self, x):
        x, _ = self.attention(x.transpose(1, 2))
        return x

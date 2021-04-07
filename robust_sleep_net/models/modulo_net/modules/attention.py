import torch
import torch.nn as nn

activation_functions = {"tanh": nn.Tanh, "relu": nn.ReLU}
import numpy as np


class Attention(nn.Module):
    """"
    Attention module similarly to Luong 2015
    """

    def __init__(self, input_dim, context_size=32, activation="tanh"):
        """

        input_dim (int): Dimensions of the input/ Number of features_data
        context_size (int): number of dim to use from the context
        """
        super(Attention, self).__init__()

        self.context_matrix = nn.Linear(input_dim, context_size)
        self.context_vector = nn.Linear(context_size, 1, bias=False)

        self.softmax = torch.nn.Softmax(dim=1)
        self.tanh = activation_functions[activation]()

    def forward(self, x):
        """

        x (tensor: batch_size,sequence length,input_dim):
        returns x (tensor: batch_size,input_dim),  attention_weights (tensor: batch_size,sequence_length)
        """
        batch_size, length, n_features = x.shape

        x_att = x.reshape(batch_size * length, n_features)
        u = self.context_matrix(x_att)
        u = self.tanh(u)
        uv = self.context_vector(u)
        uv = uv.view(batch_size, length)
        alpha = torch.nn.Softmax(dim=1)(uv)
        alpha = alpha.unsqueeze(-1)
        x_out = alpha * x
        x_out = x_out.sum(1)

        return x_out, alpha

import numpy as np
import torch
from torch import nn


class PoolReducer(nn.Module):
    def __init__(self, groups, pool_operation="max"):
        super(PoolReducer, self).__init__()
        if pool_operation == "max":
            self.pool = lambda x: x.max(2)[0]
        elif pool_operation == "average":
            self.pool = lambda x: x.mean(2)
        else:
            raise ValueError('pool_operation must be in ["max","average"]')

    def forward(self, x):
        return self.pool(x)

from torch import nn
from torch.optim import Adam, SGD

optimizers = {"adam": Adam, "sgd": SGD}

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn import Module


class CELoss(Module):
    def __init__(self, num_classes=5, weight=None):
        super(CELoss, self).__init__()
        if isinstance(weight, list):
            assert len(weight) == num_classes
            weight = torch.Tensor(np.array(weight)).float()
        if isinstance(weight, dict):
            assert len(weight) == num_classes
            weight = torch.Tensor(np.array(list(weight.values()))).float()

        self.num_classes = num_classes
        if torch.cuda.is_available():
            if weight is not None:
                self.prediction_criterion = nn.NLLLoss(weight=weight.cuda()).cuda()
            else:
                self.prediction_criterion = nn.NLLLoss().cuda()
        else:
            if weight is not None:
                self.prediction_criterion = nn.NLLLoss(weight=weight)
            else:
                self.prediction_criterion = nn.NLLLoss()

    def forward(self, x, target, smooth_dist=None):

        # Softmax and log of output
        if isinstance(x, tuple):
            pred_original, _ = x
        else:
            pred_original = x
        pred_original = F.softmax(pred_original, dim=-1)
        # Make sure we don't have any numerical instability
        eps = 1e-12
        pred = torch.clamp(pred_original, 0.0 + eps, 1.0 - eps)
        pred = torch.log(pred)
        xentropy_loss = self.prediction_criterion(pred, target)
        return xentropy_loss


loss_functions = {
    "cross_entropy": nn.CrossEntropyLoss,
    "cross_entropy_with_weights": CELoss,
}

from .trainer import Trainer

__all__ = ["Trainer"]

import torch
from torch.autograd import Function
import torch.nn.functional as F
from torch import nn
import math
from lib.normalize import Normalize
import copy

class SmoothCrossEntropy(nn.Module):
    """
    """
    def __init__(self, lambd):
        super(SmoothCrossEntropy, self).__init__()
        self.lambda = lambd

    def forward(outputs, targets, findexes, lemniscate): # indexes
        loss = nn.CrossEntropyLoss(outputs, targets)
        loss_aux = 0
        loss = loss + loss_aux
        return loss

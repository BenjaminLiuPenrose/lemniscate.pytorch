import torch
from torch.autograd import Function
import torch.nn.functional as F
from torch import nn
import math
from lib.normalize import Normalize
import copy
from pdb import set_trace as st

class SmoothCrossEntropy(nn.Module):
    """
    """
    def __init__(self, lambd):
        super(SmoothCrossEntropy, self).__init__()
        self.lambd = lambd

    def forward(self, outputs, targets, findexes, lemniscate, sample_duration): # indexes
        lambd = self.lambd
        criterion = nn.CrossEntropyLoss()
        criterion_aux = nn.MSELoss(reduction = "sum")
        st()
        loss = criterion(outputs, targets)
        vector_x = [lemniscate.vectorBank[findexes[fi], :] for fi in range(len(findexes)) if (fi+1) % sample_duration != 0 ]
        vector_y = [lemniscate.vectorBank[findexes[fi], :] for fi in range(len(findexes)) if fi % sample_duration != 0]
        vector_x = torch.FloatTensor(vector_x)
        vector_y = torch.FloatTensor(vector_y)
        loss_aux = criterion_aux(vector_x, vector_y)

        loss = loss + lambd * loss_aux
        return loss

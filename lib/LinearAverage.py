import torch
from torch.autograd import Function
import torch.nn.functional as F
from torch import nn
import math
from lib.utils import normalize

class LinearAverageOp(Function):
    @staticmethod
    def forward(self, x, y, memory, params):
        T = params[0].item()
        batchSize = x.size(0)
        momentum = params[1].item()


        # inner product
        out = torch.mm(x.data, memory.t())
        out.div_(T) # batchSize * N

        self.save_for_backward(x, memory, y, params)

        return out

    @staticmethod
    def backward(self, gradOutput):
        x, memory, y, params = self.saved_tensors
        batchSize = gradOutput.size(0)
        T = params[0].item()
        momentum = params[1].item()

        # add temperature
        gradOutput.data.div_(T)

        # gradient of linear
        gradInput = torch.mm(gradOutput.data, memory)
        gradInput.resize_as_(x)

        # # update the non-parametric data, comment here for w not equals v
        # weight_pos = memory.index_select(0, y.data.view(-1)).resize_as_(x)
        # weight_pos.mul_(momentum)
        # weight_pos.add_(torch.mul(x.data, 1-momentum))
        # w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        # updated_weight = weight_pos.div(w_norm)
        # memory.index_copy_(0, y, updated_weight)

        return gradInput, None, None, None

class LinearAverage(nn.Module):

    def __init__(self, inputSize, outputSize, T=0.07, momentum=0.5):
        super(LinearAverage, self).__init__()
        stdv = 1 / math.sqrt(inputSize)
        self.nLem = outputSize

        self.register_buffer('params',torch.tensor([T, momentum]));
        stdv = 1. / math.sqrt(inputSize/3)
        ### stop w = v process to make w stand alone
        # self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2*stdv).add_(-stdv))
        # self.register_parameter('memory', None)
        self.memory = nn.Parameter(torch.rand(outputSize, inputSize).mul_(2*stdv).add_(-stdv) )


    def forward(self, x, y):
        # print(self.memory.requires_grad)
        self.memory.t() = F.normalize(self.memory.t(), p = 2, dim = 1)
        out = LinearAverageOp.apply(x, y, self.memory, self.params)
        return out


class FeatureBankOp(Function):
    @staticmethod
    def forward(self, x, y, memory, params):
        batchSize = x.size(0)

        # inner product
        out = x

        self.save_for_backward(x, memory, y, params)

        return out

    @staticmethod
    def backward(self, gradOutput):
        x, memory, y, params = self.saved_tensors
        momentum = params[1].item()
        batchSize = gradOutput.size(0)

        # update the non-parametric data
        weight_pos = memory.index_select(0, y.data.view(-1)).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(x.data, 1-momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)

        return gradOutput, None, None, None

class FeatureBank(nn.Module):
    def __init__(self, inputSize, outputSize, T = 0.07, momentum = 0.5):
        super(FeatureBank, self).__init__()
        stdv = 1 / math.sqrt(inputSize / 3)
        self.register_buffer('params',torch.tensor([T, momentum]));
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2*stdv).add_(-stdv))
        self.nLem = outputSize
        self.momentum = momentum

    def forward(self, x, y):
        out = FeatureBankOp.apply(x, y, self.memory, self.params)
        return out

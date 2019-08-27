from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from scipy import stats

import os, sys, time, math, copy
from collections import OrderedDict
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style("darkgrid")
from pdb import set_trace as st
import warnings
warnings.filterwarnings("ignore")

### load resnet
from models.resnet import resnet18
low_dim = 128
spatial_size = 224
norm_value = 255
sample_duration = 1
video_path = './data/UCF-101-Frame/'
annotation_path = './data/UCF-101-Annotate/ucfTrainTestlist/ucf101_01.json'

resnet = resnet18(low_dim = low_dim, spatial_size = spatial_size)
checkpoint = torch.load('./checkpoint/' + 'ckpt_ucf101_test31.t7')
checkpoint2 = copy.deepcopy(checkpoint)
checkpoint2['net'] = OrderedDict([(".".join(k.split('.')[1:]), v) for k, v in checkpoint['net'].items()])
resnet.load_state_dict(checkpoint2['net'])


### prepare datasets
import datasets
import torchvision.transforms as transforms
from utils.spatial_transforms import Compose, Normalize, RandomHorizontalFlip, MultiScaleRandomCrop, ToTensor, CenterCrop
from utils.temporal_transforms import TemporalRandomCrop
import lib.custom_transforms as custom_transforms


transform_test = {
    'spatial': Compose([
        CenterCrop(spatial_size),
        ToTensor(norm_value),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]),
    'temporal': None, # TemporalRandomCrop(args.sample_duration),
    'target': None,
}
testset = datasets.UCF101Instance(
                video_path,
                annotation_path,
                'validation',
                spatial_transform = transform_test['spatial'],
                temporal_transform = transform_test['temporal'],
                target_transform = transform_test['target'],
                sample_duration = sample_duration,
                n_samples_for_each_video = 1
            )
testloader = torch.utils.data.DataLoader(
                testset,
                batch_size = int( 128 / sample_duration ),
                shuffle = False,
                num_workers = 2
            )

X = np.load("best_acc_ucf_cls_test31.npy")
y = np.load("best_acc_ucf_clsy_test31.npy")
# X = X[:, :-1]
X.shape, y.shape

### MLP
### MLP
class mlp(nn.Module):
    def __init__(self, input_size, layers, num_classes):
        super(mlp, self).__init__()
        self.fc = {}
        self.relu = {}
        self.layers = layers
        for i, hidden_size in enumerate(layers):
            self.fc[i+1] = nn.Linear(input_size, hidden_size)
            self.relu[i+1] = nn.ReLU()
            input_size = hidden_size
        self.fcl = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = x
        for i, hidden_size in enumerate(self.layers):
            out = self.fc[i+1](out)
            out = self.relu[i+1](out)
        out = self.fcl(out)
        # out = F.log_softmax(out)
        return out

### build mlp
input_size = 128
layers = [
    64,
    32
]
n_classes = 101
net = mlp(input_size, layers, n_classes)
# net.cuda()

import torch.optim as optim
learning_rate = 0.3
num_epoch = 100
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
criterion = nn.CrossEntropyLoss()

### train mlp
bsize = 128
for epoch in range(num_epoch):
    for fi in range( int(X.shape[0] / bsize) ):
        optimizer.zero_grad()  # zero the gradient buffer
        output = net( torch.tensor(X[(fi * bsize):(fi * bsize + bsize), :]) )
        st()
        loss = criterion(output, torch.tensor(y[(fi * bsize):(fi * bsize + bsize)]).view(-1))
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

### evaluate mlp
y_hat = []
with torch.no_grad():
    for batch_idx, (inputs, targets, indexes, findexes) in enumerate(testloader):
        b, d, c, w, h = inputs.shape; inputs = inputs.view(b*d, c, w, h)
        b, d = targets.shape; targets = targets.view(b*d)
        b, d = indexes.shape; indexes = indexes.view(b*d)
        b, d = findexes.shape; findexes = findexes.view(b*d)
        features = resnet(inputs)
        for fi in range(features.shape[0]):
            y_hat.append(
                net(features[fi, :])
            )

y_test = np.array(y_test)
y_test2 = np.array([y_test[i * sample_duration] for i in range(  int(len(y_test) / sample_duration)) ])

y_hat2 = []
for i in range(len(y_hat) / sample_duration):
    bg = i * sample_duration
    ed = i * sample_duration + sample_duration
    y_hat_selected = y_hat[bg: ed]
    y_hat2.append( stats.mode(y_hat_selected)[0] )

acc = accuracy_score(y_test, y_hat)
acc2 = accuracy_score(y_test2, y_hat2)
acc * 100, acc2 * 100

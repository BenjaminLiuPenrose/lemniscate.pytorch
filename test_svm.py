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
low_dim = 512
spatial_size = 224
norm_value = 255
sample_duration = 4
n_samples_for_each_video = 1
experiment_num = "41-x1"
video_path = './data/UCF-101-Frame/'
annotation_path = './data/UCF-101-Annotate/ucfTrainTestlist/ucf101_01.json'

resnet = resnet18(low_dim = low_dim, spatial_size = spatial_size)
checkpoint = torch.load('./checkpoint/' + 'ckpt_ucf101_test{}.t7'.format(experiment_num))
checkpoint2 = copy.deepcopy(checkpoint)
checkpoint2['net'] = OrderedDict([(".".join(k.split('.')[1:]), v) for k, v in checkpoint['net'].items()])
resnet.load_state_dict(checkpoint2['net'])
sample_duration = sample_duration if n_samples_for_each_video == 1 else n_samples_for_each_video

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

X = np.load("best_acc_ucf_cls_test{}.npy".format(experiment_num))
y = np.load("best_acc_ucf_clsy_test{}.npy".format(experiment_num))
# X = X[:, :-1]
y = y.repeat(n_samples_for_each_video) # repeat in the code

### SVM
def svc_param_selection(X, y, nfolds, verbose = 2):
    Cs = [0.01, 0.1, 1, 10, 100]
    gammas = [0.01, 0.1, 1, 10]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid,  cv=nfolds, verbose = 2) # scoring='accuracy',
    grid_search.fit(X, y)
    grid_search.best_params_
    if verbose > 1:
        from IPython.display import display
        display(pd.DataFrame.from_dict( grid_search.cv_results_ ))
    return grid_search.best_params_

### prepare X_test, y_test
X_test, y_test = [], []
with torch.no_grad():
    for batch_idx, (inputs, targets, indexes, findexes) in enumerate(testloader):
        b, d, c, w, h = inputs.shape; inputs = inputs.view(b*d, c, w, h)
        b, d = targets.shape; targets = targets.view(b*d)
        b, d = indexes.shape; indexes = indexes.view(b*d)
        b, d = findexes.shape; findexes = findexes.view(b*d)
        features = resnet(inputs)
#         st()
        features = features.cpu().numpy()
        targets = targets.cpu().numpy()
        for fi in range(features.shape[0]):
            X_test.append( features[fi, :] )
            y_test.append( targets[fi] )
X_test = np.array(X_test)
y_test = np.array(y_test)
y_test2 = np.array([y_test[i * sample_duration] for i in range(  int(len(y_test) / sample_duration))] )

### build and train SVM
st()
params = svc_param_selection(X, y, nfolds = 5)
clf = SVC(**params)
clf.fit(X, y);

### evaluate SVM
# X_test = X
# y_test = y
y_hat = clf.predict(X_test)
### voting of each class
y_hat2 = []
for i in range(int( len(y_hat) / sample_duration)):
    bg = i * sample_duration
    ed = i * sample_duration + sample_duration
    y_hat_selected = y_hat[bg: ed]
    y_hat2.append( stats.mode(y_hat_selected)[0] )

y_hat2 = [stats.mode(y_hat[i * sample_duration:(i * sample_duration + sample_duration)])[0] for i in range( int(len(y_hat) / sample_duration))  ]

acc = accuracy_score(y_test, y_hat)
acc2 = accuracy_score(y_test2, y_hat2)
acc * 100, acc2 * 100
st()

### in test
y_hat_t = clf.predict(X)
acc = accuracy_score(y, y_hat_t)
acc * 100

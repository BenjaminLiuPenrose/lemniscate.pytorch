'''Train UCF101 with PyTorch.'''
from __future__ import print_function
from pdb import set_trace as st

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from utils.spatial_transforms import Compose, Normalize, RandomHorizontalFlip, MultiScaleRandomCrop, ToTensor, CenterCrop
from utils.temporal_transforms import TemporalRandomCrop
import lib.custom_transforms as custom_transforms

import sys, os, time, math
import argparse
import numpy as np

import models
from models.resnet import resnet18
from models import resnet_ucf101
import datasets

from lib.NCEAverage import NCEAverage
from lib.LinearAverage import LinearAverage, LinearAverageWithWeights
from lib.NCECriterion import NCECriterion
from lib.SmoothCrossEntropy import SmoothCrossEntropy
from lib.utils import AverageMeter
from test import NN, kNN, kNN_ucf101, kNN_ucf101_store

parser = argparse.ArgumentParser(description='PyTorch UCF101 Training')
parser.add_argument('--lr', default=0.03, type=float, help='learning rate')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--low-dim', default=128, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--nce-k', default=4096, type=int,
                    metavar='K', help='number of negative samples for NCE')
parser.add_argument('--nce-t', default=0.1, type=float,
                    metavar='T', help='temperature parameter for softmax')
parser.add_argument('--nce-m', default=0.5, type=float,
                    metavar='M', help='momentum for non-parametric updates')

parser.add_argument('--video_path', '-video', default='./data/UCF-101-Frame/',
                    type=str, help='video path')
parser.add_argument('--annotation_path', '-anno',
                    default='./data/UCF-101-Annotate/UCF101_Action_detection_splits/',
                    type=str, help='annotation path')
parser.add_argument('--norm_value', default=255, type=int, help='Divide inputs by 255 or 1')
parser.add_argument('--sample_duration', default=32, type=int, help='Temporal duration of inputs')
parser.add_argument('--n_samples_for_each_video', default=1, type=int, help='Temporal duration of inputs')
parser.add_argument('--spatial_size', default=224, type=int, help='Height and width of inputs')
parser.add_argument('--initial_scale', default=1.0, type=float, help='Initial scale for multiscale cropping')
parser.add_argument('--num_scales', default=5, type=int, help='Number of scales for multiscale cropping')
parser.add_argument('--scale_step', default=0.84089641525, type=float, help='Scale step for multiscale cropping')
parser.add_argument('--resnet_shortcut', default='B', type=str, help='Shortcut type of resnet (A | B)')

args = parser.parse_args()

args.scales = [args.initial_scale]
for i in range(1, args.num_scales):
    args.scales.append(
        args.scales[-1] * args.scale_step
    )

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0
start_glob = time.time()

### Data
print('==> Preparing data..')
transform_train = {
    'spatial': Compose([
        MultiScaleRandomCrop(args.scales, args.spatial_size),
        RandomHorizontalFlip(),
        ToTensor(args.norm_value),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]),
    'temporal': None, # TemporalRandomCrop(args.sample_duration),
    'target': None,
}
transform_test = {
    'spatial': Compose([
        CenterCrop(args.spatial_size),
        ToTensor(args.norm_value),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]),
    'temporal': None, # TemporalRandomCrop(args.sample_duration),
    'target': None,
}

trainset = datasets.UCF101Instance(
                args.video_path,
                args.annotation_path,
                'training',
                spatial_transform = transform_train['spatial'],
                temporal_transform = transform_train['temporal'],
                target_transform = transform_train['target'],
                sample_duration = args.sample_duration,
                n_samples_for_each_video = args.n_samples_for_each_video
            )
trainloader = torch.utils.data.DataLoader(
                trainset,
                batch_size = int( 128 / args.sample_duration ),
                shuffle = True,
                num_workers =  2
            )

testset = datasets.UCF101Instance(
                args.video_path,
                args.annotation_path,
                'validation',
                spatial_transform = transform_test['spatial'],
                temporal_transform = transform_test['temporal'],
                target_transform = transform_test['target'],
                sample_duration = args.sample_duration,
                n_samples_for_each_video = args.n_samples_for_each_video
            )
testloader = torch.utils.data.DataLoader(
                testset,
                batch_size = int( 128 / args.sample_duration ),
                shuffle = False,
                num_workers = 2
            )
ndata = trainset.__len__()

### Build Model
print('==> Building model..')
### Define net
net = resnet18(low_dim=args.low_dim, spatial_size = args.spatial_size)
### modify 0813
# net = resnet_ucf101.resnet18(
#             num_classes = args.low_dim,
#             shortcut_type = args.resnet_shortcut,
#             spatial_size = args.spatial_size,
#             sample_duration = args.sample_duration
#         )

### Define lemniscate
if args.nce_k > 0:
    assert False
else:
    lemniscate = LinearAverageWithWeights(
        args.low_dim,
        ndata,
        args.nce_t,
        args.nce_m,
        sample_duration = args.sample_duration,
        n_samples_for_each_video = args.n_samples_for_each_video
    )

if device == 'cuda':
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count() ))
    cudnn.benchmark = True

### Test only
if args.test_only or len(args.resume) > 0:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/' + args.resume)
    net.load_state_dict(checkpoint['net'])
    lemniscate = checkpoint['lemniscate']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

### Dfine criterion/loss
if hasattr(lemniscate, 'K'):
    assert False
else:
    ### vector embedding, original
    criterion = nn.CrossEntropyLoss()
    ### smooth loss
    # criterion = SmoothCrossEntropy(lambd = .01)

net.to(device)
lemniscate.to(device)
criterion.to(device)

if args.test_only:
    acc, acc_top5 = kNN(0, net, lemniscate, trainloader, testloader, 200, args.nce_t, 1)
    sys.exit(0)

optimizer = optim.SGD(list(net.parameters())+ list(lemniscate.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr
    if epoch >= 80:
        lr = args.lr * (0.1 ** ((epoch-80) // 40))
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    net.train()
    end = time.time()
    for batch_idx, (inputs, targets, indexes, findexes) in enumerate(trainloader):
        # if batch_idx > 3:
        #     return
        data_time.update(time.time() - end)
        # view targets
        inputs, targets, indexes, findexes = inputs.to(device), targets.to(device), indexes.to(device), findexes.to(device)
        ### modify 0813
        b, d, c, w, h = inputs.shape; inputs = inputs.view(b*d, c, w, h)
        b, d = targets.shape; targets = targets.view(b*d)
        b, d = indexes.shape; indexes = indexes.view(b*d)
        b, d = findexes.shape; findexes = findexes.view(b*d)
        ### modify 0813
        optimizer.zero_grad()
        # st()
        features = net(inputs)

        ### vector embedding, original, smooth loss
        outputs = lemniscate(features, indexes, findexes)
        ### smooth loss
        # outputs = lemniscate(features, indexes, findexes)
        ### vector embedding, original
        loss = criterion(outputs, indexes)
        ### smooth loss
        # loss = criterion(outputs, indexes, findexes, lemniscate, args.sample_duration)

        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), inputs.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        print('Epoch: [{}][{}/{}]'
              'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
              'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
              'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})'.format(
              epoch, batch_idx, len(trainloader), batch_time=batch_time, data_time=data_time, train_loss=train_loss))

for epoch in range(start_epoch, start_epoch + 100):
    train(epoch)
    acc, acc_top5 = kNN_ucf101(epoch, net, lemniscate, trainloader, testloader, 200, args.nce_t, 0)
    # if epoch > 3:
    #     break
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'lemniscate': lemniscate,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_ucf101_test52-x1.t7')
        best_acc = acc
        print("="*100+"saving best_acc_ucf.npy"+"="*100)
        ### modify 0814
        # X = np.append(lemniscate.memory.cpu().detach().numpy(), np.array([trainset.targets]).T, axis = 1)
        X = lemniscate.vectorBank.cpu().detach().numpy()
        # X = lemniscate.memory.cpu().detach().numpy()
        repeat_size = args.sample_duration if args.n_samples_for_each_video == 1 else args.n_samples_for_each_video
        y = np.array([trainset.targets]).repeat(repeat_size).T
        np.save("best_acc_ucf_cls_test52-x1.npy", X)
        np.save("best_acc_ucf_clsy_test52-x1.npy", y)
        X_test, y_test = kNN_ucf101_store(epoch, net, lemniscate, trainloader, testloader, 200, args.nce_t, 0)
        np.save("best_acc_ucf_clst_test52-x1.npy", X_test)
        np.save("best_acc_ucf_clsyt_test52-x1.npy", y_test)
        ### modify 0814
    print('best accuracy: {:.2f}'.format(best_acc*100))

acc, acc_top5 = kNN_ucf101(0, net, lemniscate, trainloader, testloader, 200, args.nce_t, 0)
print('last accuracy: {:.2f}'.format(acc*100))
end_glob = time.time()
print('total Training time: {}'.format(end_glob - start_glob) )

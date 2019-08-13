'''Train UCF101 with PyTorch.'''
from __future__ import print_function
from pdb import set_trace as st

import sys
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

import os
import argparse
import time
import numpy as np

import models
from models import resnet_ucf101
import datasets
import math

from lib.NCEAverage import NCEAverage
from lib.LinearAverage import LinearAverage, LinearAverageWithWeights
from lib.NCECriterion import NCECriterion
from lib.utils import AverageMeter
from test import NN, kNN, kNN_ucf101
# from tensorboardX import SummaryWriter

########
### add SummaryWriter
#######
#######
### add gpu device
#######
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
if not os.path.exists('./checkpoint'):
    os.mkdir('./checkpoint')
# writer = SummaryWriter(logdir = './checkpoint', comment = 'UCF101')
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
parser.add_argument('--spatial_size', default=224, type=int, help='Height and width of inputs')
parser.add_argument('--initial_scale', default=1.0, type=float, help='Initial scale for multiscale cropping')
parser.add_argument('--num_scales', default=5, type=int, help='Number of scales for multiscale cropping')
parser.add_argument('--scale_step', default=0.84089641525, type=float, help='Scale step for multiscale cropping')
parser.add_argument('--resnet_shortcut', default='B', type=str, help='Shortcut type of resnet (A | B)')

args = parser.parse_args()

args.scales = [args.initial_scale]
for i in range(1, args.num_scales):
    args.scales.append(args.scales[-1] * args.scale_step)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# Data
start_glob = time.time()

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.CenterCrop(size=args.spatial_size),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomGrayscale(p=0.2),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(size=args.spatial_size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = datasets.UCF101Instance(
            args.video_path,
            args.annotation_path,
            'training',
            transform = transform_train,
            n_samples_for_each_video = 1,
            sample_duration = args.sample_duration
            )
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=int(128 / args.sample_duration), shuffle=True, num_workers=2)
# trainset = datasets.CIFAR100Instance(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = datasets.UCF101Instance(
            args.video_path,
            args.annotation_path,
            'validation',
            transform = transform_test,
            n_samples_for_each_video = 1,
            sample_duration = args.sample_duration
            )
# testloader = torch.utils.data.DataLoader(testset, batch_size=int(128 / args.sample_duration), shuffle=False, num_workers=2)
# testset = datasets.CIFAR100Instance(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

ndata = trainset.__len__()

print('==> Building model..')
net = models.__dict__['ResNet18'](low_dim=args.low_dim)
### ROLLBACK
# net = resnet_ucf101.resnet18(
#                 num_classes=args.low_dim,
#                 shortcut_type=args.resnet_shortcut,
#                 spatial_size=args.spatial_size,
#                 sample_duration=1 #args.sample_duration
# )

# define leminiscate
if args.nce_k > 0:
    lemniscate = NCEAverage(args.low_dim, ndata, args.nce_k, args.nce_t, args.nce_m)
else:
    lemniscate = LinearAverageWithWeights(args.low_dim, ndata, args.nce_t, args.nce_m)

if device == 'cuda':
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

# Model
if args.test_only or len(args.resume)>0:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+args.resume)
    net.load_state_dict(checkpoint['net'])
    lemniscate = checkpoint['lemniscate']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# define loss function
if hasattr(lemniscate, 'K'):
    criterion = NCECriterion(ndata)
else:
    criterion = nn.CrossEntropyLoss()

net.to(device)
lemniscate.to(device)
criterion.to(device)

if args.test_only:
    acc = kNN(0, net, lemniscate, trainloader, testloader, 200, args.nce_t, 1)
    sys.exit(0)

# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer = optim.SGD(list(net.parameters())+list(lemniscate.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)


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

    # switch to train mode
    net.train()

    end = time.time()
    for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
        data_time.update(time.time() - end)
        inputs, targets, indexes = inputs.to(device), targets.to(device), indexes.to(device)
        # st()
        # print("="*50, inputs.shape, targets.shape, indexes.shape)
        optimizer.zero_grad()

        features = net(inputs)
        outputs = lemniscate(features, indexes)
        loss = criterion(outputs, indexes)

        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('Epoch: [{}][{}/{}]'
              'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
              'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
              'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})'.format(
              epoch, batch_idx, len(trainloader), batch_time=batch_time, data_time=data_time, train_loss=train_loss))



for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    acc, acc_top5 = kNN_ucf101(epoch, net, lemniscate, trainloader, testloader, 200, args.nce_t, 0)
    # writer.add_scalar('data/acc', acc, epoch)
    # writer.add_scalar('data/acc_top5', acc_top5, epoch)

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
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc
        print("="*100+"saving best_acc_ucf.npy"+"="*100)
        X = np.append(lemniscate.memory.cpu().detach().numpy(), np.array([trainset.targets]).T, axis = 1)
        np.save("best_acc_ucf.npy", X)

    print('best accuracy: {:.2f}'.format(best_acc*100))
# writer.export_scalars_to_json("./all_scalars.json")
# writer.close()

acc = kNN_ucf101(0, net, lemniscate, trainloader, testloader, 200, args.nce_t, 1)
print('last accuracy: {:.2f}'.format(acc*100))

end_glob = time.time()
print('total Training time: {}'.format(end_glob - start_glob) )

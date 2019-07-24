'''Train CIFAR100 with PyTorch.'''
from __future__ import print_function

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import lib.custom_transforms as custom_transforms

import os
import argparse
import time

import models
import datasets
import math
import numpy as np

from lib.NCEAverage import NCEAverage
from lib.LinearAverage import LinearAverage, FeatureBank
from lib.NCECriterion import NCECriterion
from lib.utils import AverageMeter, normalize
from test import NN, kNN

from utils.datasets import BalancedBatchSampler_CIFAR
from utils.losses import OnlineContrastiveLoss
from utils.utils import AllPositivePairSelector, HardNegativePairSelector, AllNegativePairSelector # Strategies for selecting pairs within a minibatch
from utils.metrics import AccumulatedAccuracyMetric

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
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
parser.add_argument('--margin', default=.1, type=float,
                    help='margin for Saimese loss')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
start_glob = time.time()

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.2,1.)),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomGrayscale(p=0.2),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = datasets.CIFAR10Instance(root='./data', train=True, download=True, transform=transform_train)
trainset.init()
# trainset = datasets.CIFAR100Instance(root='./data', train=True, download=True, transform=transform_train)
train_batch_sampler = BalancedBatchSampler_CIFAR(trainset.indices, n_classes=10*25, n_samples=1)
# trainloader = torch.utils.data.DataLoader(trainset, batch_sampler = train_batch_sampler, num_workers=2, pin_memory = True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128*2, shuffle=True, num_workers=2)

testset = datasets.CIFAR10Instance(root='./data', train=False, download=True, transform=transform_test)
testset.init()
# testset = datasets.CIFAR100Instance(root='./data', train=False, download=True, transform=transform_test)
test_batch_sampler = BalancedBatchSampler_CIFAR(testset.indices, n_classes=10*25, n_samples=1)
# testloader = torch.utils.data.DataLoader(testset, batch_sampler = test_batch_sampler, num_workers=2, pin_memory = True)
testloader = torch.utils.data.DataLoader(testset, batch_size=100*2, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
ndata = trainset.__len__()

print('==> Building model..')
net = models.__dict__['ResNet18'](low_dim=args.low_dim)
# define lemniscate
if args.nce_k > 0:
    assert False
    lemniscate = NCEAverage(args.low_dim, ndata, args.nce_k, args.nce_t, args.nce_m)
else:
    # lemniscate = LinearAverage(args.low_dim, ndata, args.nce_t, args.nce_m)
    lemniscate = FeatureBank(args.low_dim, ndata, args.nce_t, args.nce_m)
metrics = []

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
    assert False
    criterion = NCECriterion(ndata)
else:
    # criterion = nn.CrossEntropyLoss()
    criterion = OnlineContrastiveLoss(args.margin, AllNegativePairSelector())

net.to(device)
lemniscate.to(device)
criterion.to(device)

if args.test_only:
    acc = kNN(0, net, lemniscate, trainloader, testloader, 200, args.nce_t, 1, async_bank = True)
    sys.exit(0)

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# optimizer = optim.SGD(list(net.parameters())+list(lemniscate.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr
    if epoch >= 80:
        lr = args.lr * (0.1 ** ((epoch-80) // 40))
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# def train(epoch):
#     print('\nEpoch: %d' % epoch)
#     adjust_learning_rate(optimizer, epoch)
#     train_loss = AverageMeter()
#     data_time = AverageMeter()
#     batch_time = AverageMeter()
#     correct = 0
#     total = 0
#
#     net.train()
#
#     end = time.time()
#     for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
#         data_time.update(time.time() - end)
#         inputs, targets, indexes = inputs.to(device), targets.to(device), indexes.to(device)
#
#         optimizer.zero_grad()
#         features = net(inputs)
#         loss = criterion(features, indexes)
#
#         loss.backward()
#         optimizer.step()
#         train_loss.update(loss.item(), inputs.size(0))
#
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#         msg = ('Epoch: [{}][{}/{}]'
#               'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
#               'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
#               'Loss: {train_loss.val:.3f} ({train_loss.avg:.3f})'.format(
#               epoch, batch_idx, len(trainloader), batch_time=batch_time, data_time=data_time, train_loss=train_loss))
#         for metric in metrics:
#             msg += '\t{}: {}'.format(metric.name(), metric.value())
#         print(msg)
#
# metric = AccumulatedAccuracyMetric()
# def test(epoch):
#     with torch.no_grad():
#         net.eval()
#         metric.reset()
#         test_loss = 0
#         for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
#             inputs, targets, indexes = inputs.to(device), targets.to(device), indexes.to(device)
#             features = net(inputs)
#             loss = criterion(features, indexes)
#             test_loss += loss.item()
#             acc = metric(features, indexes, loss)
#         return test_loss / len(testloader), acc
#
# for epoch in range(start_epoch, start_epoch+200):
#     train(epoch)
#     test_loss, acc = test(epoch)
#
#     if acc > best_acc:
#         print('Saving..')
#         state = {
#             'net': net.state_dict(),
#             'lemniscate': lemniscate,
#             'acc': acc,
#             'epoch': epoch,
#         }
#         if not os.path.isdir('checkpoint'):
#             os.mkdir('checkpoint')
#         torch.save(state, './checkpoint/ckpt.t7')
#         best_acc = acc
#
#     print('test_loss: {:4f}; best accuracy: {:.2f}'.format(test_loss, best_acc*100))

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    myCriterion = nn.CrossEntropyLoss()
    myLemniscate = LinearAverage(args.low_dim, ndata, args.nce_t, args.nce_m)
    train_myLoss = AverageMeter()

    # switch to train mode
    net.train()

    end = time.time()
    for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
        data_time.update(time.time() - end)
        inputs, targets, indexes = inputs.to(device), targets.to(device), indexes.to(device)
        optimizer.zero_grad()

        features = net(inputs)
        outputs = lemniscate(features, indexes)
        loss = criterion(outputs, indexes)

        with torch.no_grad():
            os = myLemniscate(features, indexes)
            myLoss = myCriterion(os, indexes)
            train_myLoss.update(myLoss.item(), inputs.size(0))

        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        msg = ('Epoch: [{}][{}/{}]'
              'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
              'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
              'Loss: {train_loss.val:.3f} ({train_loss.avg:.3f})'
              'mylos: {train_myLoss.val:.4f} ({train_myLoss.avg:.4f})'.format(
              epoch, batch_idx, len(trainloader), batch_time=batch_time, data_time=data_time, train_loss=train_loss, train_myLoss=train_myLoss))
        for metric in metrics:
            msg += '\t{}: {}'.format(metric.name(), metric.value())
        print(msg)

debug_ls =  []
for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    acc = kNN(epoch, net, lemniscate, trainloader, testloader, 200, args.nce_t, 0, async_bank = True)

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
        print("="*100+"saving best_acc.npy"+"="*100)
        np.save("best_acc.npy", lemniscate.memory.cpu())

    print('best accuracy: {:.2f}'.format(best_acc*100))

    # if epoch == start_epoch:
    #     memory_diff = lemniscate.memory.t()
    # else:
    #     memory_diff = lemniscate.memory.t() - debug_ls[-1]
    # print(memory_diff[np.nonzero()])
    # debug_ls.append(memory_diff)


acc = kNN(0, net, lemniscate, trainloader, testloader, 200, args.nce_t, 1, async_bank = True)
print('last accuracy: {:.2f}'.format(acc*100))

end_glob = time.time()
print('total Training time: {}'.format(end_glob - start_glob) )

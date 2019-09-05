import torch
import time
import datasets
from lib.utils import AverageMeter, normalize
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
from utils.losses import OnlineContrastiveLoss
from utils.utils import AllPositivePairSelector, HardNegativePairSelector, AllNegativePairSelector # Strategies for selecting pairs within a minibatch
import math
from pdb import set_trace as st

def NN(epoch, net, lemniscate, trainloader, testloader, recompute_memory=0, async_bank = False):
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    losses = AverageMeter()
    correct = 0.
    total = 0
    testsize = testloader.dataset.__len__()

    trainFeatures = lemniscate.memory.t().cuda()
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
    else:
        trainLabels = torch.LongTensor(trainloader.dataset.targets).cuda()

    if recompute_memory:
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=1)
        for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
            targets = targets.cuda(non_blocking=True)
            batchSize = inputs.size(0)
            features = net(inputs)
            # stop w = v process to make w stand alone
            trainFeatures[:, batch_idx*batchSize:batch_idx*batchSize+batchSize] = features.data.t().cuda()
        trainLabels = torch.LongTensor(temploader.dataset.targets).cuda()
        trainloader.dataset.transform = transform_bak

    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            targets = targets.cuda(non_blocking=True)
            batchSize = inputs.size(0)
            features = net(inputs)
            net_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, trainFeatures)

            yd, yi = dist.topk(1, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval = retrieval.narrow(1, 0, 1).clone().view(-1)
            yd = yd.narrow(1, 0, 1)

            total += targets.size(0)
            correct += retrieval.eq(targets.data).sum().item()

            cls_time.update(time.time() - end)
            end = time.time()

            print('Test [{}/{}]\t'
                  'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
                  'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
                  'Top1: {:.2f}'.format(
                  total, testsize, correct*100./total, net_time=net_time, cls_time=cls_time))

    return correct/total

def kNN(epoch, net, lemniscate, trainloader, testloader, K, sigma, recompute_memory=0, async_bank = False):
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    total = 0
    testsize = testloader.dataset.__len__()

    trainFeatures = lemniscate.memory.t()
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
    else:
        trainLabels = torch.LongTensor(trainloader.dataset.targets).cuda()
    C = trainLabels.max() + 1

    if recompute_memory:
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=1)
        for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
            targets = targets.cuda(non_blocking=True)
            batchSize = inputs.size(0)
            features = net(inputs)
            # stop w = v process to make w stand alone
            trainFeatures[:, batch_idx*batchSize:batch_idx*batchSize+batchSize] = features.data.t()
        trainLabels = torch.LongTensor(temploader.dataset.targets).cuda()
        trainloader.dataset.transform = transform_bak

    top1 = 0.
    top5 = 0.
    end = time.time()
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).cuda()
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            end = time.time()
            targets = targets.cuda(non_blocking=True)
            batchSize = inputs.size(0)
            features = net(inputs)
            net_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, trainFeatures)

            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            # st()
            retrieval = torch.gather(candidates, 1, yi)

            # print("+"*100, retrieval_one_hot.shape, batchSize * K, K, C)
            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions = probs.sort(1, True)

            # Find which predictions match the target
            correct = predictions.eq(targets.data.view(-1,1))

            if batch_idx == len(testloader) - 1:
                x = retrieval
                norm = x.pow(2).sum(1, keepdim = True).pow(1./2)
                print("norm of memory bank ", [n.item() for n in norm][:5] )
                x = features
                norm = x.pow(2).sum(1, keepdim = True).pow(1./2)
                print("norm of feature vector ", [n.item() for n in norm][:5] )
                # print("="*50, predictions, predictions.shape)
                # print("="*50, targets, targets.shape)
                # print(correct)

            cls_time.update(time.time() - end)

            top1 = top1 + correct.narrow(1,0,1).sum().item()
            top5 = top5 + correct.narrow(1,0,5).sum().item()

            total += targets.size(0)

            # print(predictions.)

            print('Test [{}/{}]\t'
                  'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
                  'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
                  'Top1: {:.2f}  Top5: {:.2f}'.format(
                  total, testsize, top1*100./total, top5*100./total, net_time=net_time, cls_time=cls_time))


    print(top1*100./(total + 1e-8), total, top1 )

    return top1/(total + 1e-8)


def kNN_ucf101_old(epoch, net, lemniscate, trainloader, testloader, K, sigma, recompute_memory=0, async_bank = False):
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    total = 0
    testsize = testloader.dataset.__len__()

    trainFeatures = lemniscate.memory.t()
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
    else:
        trainLabels = torch.LongTensor(trainloader.dataset.targets).cuda()
    C = trainLabels.max() + 1

    if recompute_memory:
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=1)
        for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
            targets = targets.cuda(non_blocking=True)
            batchSize = inputs.size(0)
            features = net(inputs)
            # stop w = v process to make w stand alone
            trainFeatures[:, batch_idx*batchSize:batch_idx*batchSize+batchSize] = features.data.t()
        trainLabels = torch.LongTensor(temploader.dataset.targets).cuda()
        trainloader.dataset.transform = transform_bak

    top1 = 0.
    top5 = 0.
    end = time.time()
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).cuda()

        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            end = time.time()
            targets = targets.cuda(non_blocking=True)

            # inputs = inputs.view(inputs.shape[0] * inputs.shape[2], inputs.shape[1], 1, inputs.shape[3], inputs.shape[4])
            # targets = targets.view(targets.shape[0] * targets.shape[1] )
            # indexes = indexes.view(indexes.shape[0] * indexes.shape[1] )

            batchSize = inputs.size(0)
            # st()
            features = net(inputs)
            net_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, trainFeatures)
            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            # st()
            retrieval = torch.gather(candidates, 1, yi)
            retrieval_one_hot.resize_(batchSize * K, C ).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions = probs.sort(1, True)

            # Find which predictions match the target
            correct = predictions.eq(targets.data.view(-1,1))
            if batch_idx == len(testloader) - 1:
                x = retrieval
                norm = x.pow(2).sum(1, keepdim = True).pow(1./2)
                print("norm of memory bank ", [n.item() for n in norm][:5] )
                x = features
                norm = x.pow(2).sum(1, keepdim = True).pow(1./2)
                print("norm of feature vector ", [n.item() for n in norm][:5] )
                # print("="*50, predictions, predictions.shape)
                # print("="*50, targets, targets.shape)
                # st()
            cls_time.update(time.time() - end)

            top1 = top1 + correct.narrow(1,0,1).sum().item()
            top5 = top5 + correct.narrow(1,0,5).sum().item()

            total += targets.size(0)

            print('Test [{}/{}]\t'
                  'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
                  'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
                  'Top1: {:.2f}  Top5: {:.2f}'.format(
                  total, testsize, top1*100./total, top5*100./total, net_time=net_time, cls_time=cls_time))


    print(top1*100./(total + 1e-8), total, top1 )
    print(top1/(total + 1e-8))

    return top1/(total + 1e-8), top5/(total + 1e-8)

def kNN_ucf101(epoch, net, lemniscate, trainloader, testloader, K, sigma, recompute_memory=0, async_bank = False):
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    total = 0
    testsize = testloader.dataset.__len__()

    trainFeatures = lemniscate.memory.t()
    if hasattr(trainloader.dataset, 'imgs'):
        trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
    else:
        trainLabels = torch.LongTensor(trainloader.dataset.targets).cuda()
    C = trainLabels.max() + 1

    if recompute_memory:
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=int(128 / lemniscate.sample_duration), shuffle=False, num_workers=2)
        for batch_idx, (inputs, targets, indexes, findexes) in enumerate(temploader):
            targets = targets.cuda(non_blocking=True)
            batchSize = inputs.size(0)

            b, d, c, w, h = inputs.shape; inputs = inputs.view(b*d, c, w, h)
            b, d = targets.shape; targets = targets.view(b*d)
            b, d = indexes.shape; indexes = indexes.view(b*d)
            b, d = findexes.shape; findexes = findexes.view(b*d)

            features = net(inputs)
            # stop w = v process to make w stand alone
            trainFeatures[:, batch_idx*batchSize:batch_idx*batchSize+batchSize] = features.data.t()
        trainLabels = torch.LongTensor(temploader.dataset.targets).cuda()
        trainloader.dataset.transform = transform_bak

    top1 = 0.
    top5 = 0.
    end = time.time()
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).cuda()

        for batch_idx, (inputs, targets, indexes, findexes) in enumerate(testloader):
            end = time.time()
            targets = targets.cuda(non_blocking=True)

            ### modify 0813
            b, d, c, w, h = inputs.shape; inputs = inputs.view(b*d, c, w, h)
            b, d = targets.shape; targets = targets.view(b*d)
            b, d = indexes.shape; indexes = indexes.view(b*d)
            b, d = findexes.shape; findexes = findexes.view(b*d)
            ### modify 0813

            batchSize = inputs.size(0)
            # st()
            features = net(inputs)
            net_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, trainFeatures)
            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            ### original, vector embedding
            candidates = trainLabels.repeat(lemniscate.n_samples_for_each_video).view(1,-1).expand(batchSize, -1)
            # st()
            retrieval = torch.gather(candidates, 1, yi)
            retrieval_one_hot.resize_(batchSize * K, C ).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()
            # st()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions = probs.sort(1, True)

            # of sample duration to vote
            # st()
            sample_duration = lemniscate.sample_duration if lemniscate.n_samples_for_each_video == 1 else lemniscate.n_samples_for_each_video
            predictions_reshape =  predictions.view(sample_duration, int(batchSize/sample_duration), -1).narrow(2, 0, 1).view(sample_duration, int(batchSize/sample_duration))
            predictions_vote, _ = torch.mode(predictions_reshape, dim = 0)
            targets_vote, _ = torch.mode(targets.view(sample_duration, int(batchSize/sample_duration)), dim = 0)

            # Find which predictions match the target
            correct = predictions_vote.eq(targets_vote)


            if batch_idx == len(testloader) - 1:
                x = retrieval
                norm = x.pow(2).sum(1, keepdim = True).pow(1./2)
                print("norm of memory bank ", [n.item() for n in norm][:5] )
                x = features
                norm = x.pow(2).sum(1, keepdim = True).pow(1./2)
                print("norm of feature vector ", [n.item() for n in norm][:5] )
                # print("="*50, predictions, predictions.shape)
                # print("="*50, targets, targets.shape)
                # st()
            cls_time.update(time.time() - end)

            top1 = top1 + correct.sum().item()
            top5 = top5 + 0.

            total += targets_vote.size(0)

            print('Test [{}/{}]\t'
                  'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
                  'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
                  'Top1: {:.2f}  Top5: {:.2f}'.format(
                  total, testsize, top1*100./total, top5*100./total, net_time=net_time, cls_time=cls_time))


    print(top1*100./(total + 1e-8), total, top1 )
    print(top1/(total + 1e-8))

    return top1/(total + 1e-8), top5/(total + 1e-8)

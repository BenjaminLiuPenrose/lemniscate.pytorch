import torch
import time
import datasets
from lib.utils import AverageMeter, normalize
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F

def NN(epoch, net, lemniscate, trainloader, testloader, recompute_memory=0, async_bank = False):
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    losses = AverageMeter()
    correct = 0.
    total = 0
    testsize = testloader.dataset.__len__()

    trainFeatures = lemniscate.memory.t()
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
            trainFeatures[:, batch_idx*batchSize:batch_idx*batchSize+batchSize] = features.data.t()
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
    # x = trainFeatures[:, 1]
    # norm = x.pow(2).sum().pow(1./2)
    # print("norm of sample vector ====================================================== ", norm.item())

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
            retrieval = torch.gather(candidates, 1, yi)
            if batch_idx == len(testloader) - 1:
                x = retrieval
                norm = x.pow(2).sum(1, keepdim = True).pow(1./2)
                print("norm of memory bank ", [n.item() for n in norm][:5] )
                x = features
                norm = x.pow(2).sum(1, keepdim = True).pow(1./2)
                print("norm of feature vector ", [n.item() for n in norm][:5] )
                batchSize = features.size(0)
                embeddingsDim = features.size(1)
                negative_loss = F.relu(
                    torch.bmm(
                        features.view(batchSize, 1, embeddingsDim),
                        features.view(batchSize, embeddingsDim, 1)
                        ) - 0.1
                ).mean()
                mms = torch.bmm(
                    features.view(batchSize, 1, embeddingsDim),
                    features.view(batchSize, embeddingsDim, 1)
                    ).mean()
                print("="*30)
                print("my loss ================== ", mms.item() * 1000, " ", negative_loss.item() * 1000)
                print("="*30)

            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions = probs.sort(1, True)

            # Find which predictions match the target
            correct = predictions.eq(targets.data.view(-1,1))
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
    batchSize = trainFeatures.size(0)
    embeddingsDim = trainFeatures.size(1)
    negative_loss = F.relu(
        torch.bmm(
            trainFeatures[:, :100].view(batchSize, 1, embeddingsDim),
            trainFeatures[:, -100:].view(batchSize, embeddingsDim, 1)
            ) - 0.1
    ).mean()
    mms = torch.bmm(
        trainFeatures[:, :100].view(batchSize, 1, embeddingsDim),
        trainFeatures[:, -100:].view(batchSize, embeddingsDim, 1)
        ).mean()
    print("="*30)
    print("my loss all train ========= ", mms.item() , " ", negative_loss.item() )
    print("="*30)

    print(top1*100./(total + 1e-8), total, top1 )

    return top1/(total + 1e-8)

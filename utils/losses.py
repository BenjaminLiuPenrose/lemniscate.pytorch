import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, targets):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, targets)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        # TODO
        # positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        # negative_loss = F.relu(
        #     self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
        #         1).sqrt()).pow(2)
        # print( embeddings[negative_pairs[:, 0]].shape, embeddings.shape, embeddings[negative_pairs[:, 0]][0].shape  )
        batchSize = embeddings[negative_pairs[:, 0]].size(0)
        embeddingsDim = embeddings[negative_pairs[:, 0]].size(1)
        margin = self.margin

        # square_pred = torch.pow(embeddings, 2)
        # margin_square = torch.pow(F.relu(
        #             embeddings - margin
        #         ),
        #     2)
        # positive_loss = torch.Tensor( [0] ).cuda()
        negative_loss = F.relu(
            torch.bmm(
                embeddings[negative_pairs[:, 0]].view(batchSize, 1, embeddingsDim),
                embeddings[negative_pairs[:, 1]].view(batchSize, embeddingsDim, 1)
                ) - margin
        ).pow(2)

        x = torch.bmm(
                        embeddings[negative_pairs[:, 0]].view(batchSize, 1, embeddingsDim),
                        embeddings[negative_pairs[:, 1]].view(batchSize, embeddingsDim, 1)
                        )
        norm = x.pow(2).sum(1, keepdim = True).pow(1./2)
        # print("norm of memory bank ", [n.item() for n in norm][:5], "shape ", x.shape)
        # mms = torch.bmm(
        #     embeddings[negative_pairs[:, 0]].view(batchSize, 1, embeddingsDim),
        #     embeddings[negative_pairs[:, 1]].view(batchSize, embeddingsDim, 1)
        #     ).mean()
        # print("loss bmm ", mms.item() * 1000 )

        # print(negative_loss)
        # loss = torch.cat([positive_loss, negative_loss], dim=0)
        loss = negative_loss
        print("loss dim", loss.shape, " loss mean", loss.mean().shape)
        # loss = (targets * square_pred + (1 - targets) * margin_square)
        return torch.mul(loss.mean(), 1000)


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)

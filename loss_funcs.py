import torch
import torch.nn as nn
import torch.nn.functional as F


class EuclideanContrastiveLoss(nn.Module):
    def __init__(self, margin=5.0):
        super(EuclideanContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):

        euclidean_distance = F.pairwise_distance(output1, output2)

        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss_contrastive


class CosineContrastiveLoss(nn.Module):
    def __init__(self, beta=5.0):
        super(CosineContrastiveLoss, self).__init__()
        self.beta = beta

    def forward(self, output1, output2, label):

        cosine_similarity = F.cosine_similarity(output1, output2)

        loss_cosine = torch.mean(
            -label * self.beta * cosine_similarity +
            (1 - label) * self.beta * torch.clamp(cosine_similarity, min=0)
        )

        return loss_cosine


import torch
import torch.nn as nn


class MultiplicativeLoss(nn.Module):
    def __init__(self, num_modalities=2):
        super(MultiplicativeLoss, self).__init__()
        self.num_modalities = num_modalities

    def forward(self, predictions):

        if not isinstance(predictions, torch.Tensor):
            predictions = torch.stack(predictions)

        if predictions.dim() == 3:
            pass
        elif predictions.dim() == 2:
            predictions = predictions.unsqueeze(0)
        else:
            raise ValueError("Unexpected shape for predictions.")

        n = self.num_modalities
        batch_size = predictions.size(1)
        num_classes = predictions.size(2)
        loss = 0
        for i in range(n):
            p_i = predictions[i]
            term = torch.pow(p_i, 1 / (n - 1)) * torch.log(p_i + 1e-8)
            loss -= torch.sum(term) / batch_size
        return loss


class DiscreteLoss(nn.Module):
    ''' Class to measure loss between categorical emotion predictions and labels.'''

    def __init__(self, weight_type='mean', device=torch.device('cpu')):
        super(DiscreteLoss, self).__init__()
        self.weight_type = weight_type
        self.device = device
        if self.weight_type == 'mean':
            self.weights = torch.ones((1, 26)) / 26.0
            self.weights = self.weights.to(self.device)
        elif self.weight_type == 'static':
            self.weights = torch.FloatTensor([0.1435, 0.1870, 0.1692, 0.1165, 0.1949, 0.1204, 0.1728, 0.1372, 0.1620,
                                              0.1540, 0.1987, 0.1057, 0.1482, 0.1192, 0.1590, 0.1929, 0.1158, 0.1907,
                                              0.1345, 0.1307, 0.1665, 0.1698, 0.1797, 0.1657, 0.1520,
                                              0.1537]).unsqueeze(0)
            self.weights = self.weights.to(self.device)

    def forward(self, pred, target):
        if self.weight_type == 'dynamic':
            self.weights = self.prepare_dynamic_weights(target)
            self.weights = self.weights.to(self.device)
        loss = (((pred - target) ** 2) * self.weights)
        return loss.sum()

    def prepare_dynamic_weights(self, target):
        target_stats = torch.sum(target, dim=0).float().unsqueeze(dim=0).cpu()
        weights = torch.zeros((1, 26))
        weights[target_stats != 0] = 1.0 / torch.log(target_stats[target_stats != 0].data + 1.2)
        weights[target_stats == 0] = 0.0001
        return weights
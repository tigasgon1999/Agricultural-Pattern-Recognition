import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch 
from catalyst.contrib.nn import \
    DiceLoss, IoULoss, LovaszLossMultiLabel, FocalLossBinary

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

# ACW Loss
class ACW_loss(nn.Module):
    def __init__(self,  ini_weight=0, ini_iteration=0, eps=1e-5, ignore_index=255):
        super(ACW_loss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = ini_weight
        self.itr = ini_iteration
        self.eps = eps

    def forward(self, prediction, target):
        """
        pred :    shape (N, C, H, W)
        target :  shape (N, H, W) ground truth
        return:  loss_acw
        """
        pred = F.softmax(prediction, 1)
        one_hot_label, mask = self.encode_one_hot_label(pred, target)

        acw = self.adaptive_class_weight(pred, one_hot_label, mask)

        # Positive and negative class balanced function (PNC)
        err = torch.pow((one_hot_label - pred), 2)
        
        # one = torch.ones_like(err)
        pnc = err - ((1. - err + self.eps) / (1. + err + self.eps)).log()
        loss_pnc = torch.sum(acw * pnc, 1)


        intersection = 2 * torch.sum(pred * one_hot_label, dim=(0, 2, 3)) + self.eps
        union = pred + one_hot_label

        if mask is not None:
            union[mask] = 0

        union = torch.sum(union, dim=(0, 2, 3)) + self.eps
        dice = intersection / union

        return loss_pnc.mean() - dice.mean().log()

    def adaptive_class_weight(self, pred, one_hot_label, mask=None):
        self.itr += 1

        sum_class = torch.sum(one_hot_label, dim=(0, 2, 3))
        sum_norm = sum_class / sum_class.sum()
        
        # Pixel frequency of class j
        self.weight = (self.weight * (self.itr - 1) + sum_norm) / self.itr
        
        # Iterative median frequency class weight
        mfb = self.weight.mean() / (self.weight + self.eps)

        # Normalised iterative weights with adaptive broadcasting
        mfb = mfb / mfb.sum()
        acw = (1. + pred + one_hot_label) * mfb.unsqueeze(-1).unsqueeze(-1)

        if mask is not None:
            acw[mask] = 0

        return acw

    def encode_one_hot_label(self, pred, target):
        one_hot_label = pred.detach() * 0
        if self.ignore_index is not None:
            mask = (target == self.ignore_index)
            target = target.clone()
            target[mask] = 0
            one_hot_label.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(one_hot_label)
            one_hot_label[mask] = 0
            return one_hot_label, mask
        else:
            one_hot_label.scatter_(1, target.unsqueeze(1), 1)
            return one_hot_label, None

# Composed Loss
str2Loss = {
    "bce": nn.BCEWithLogitsLoss,
    "dice": DiceLoss,
    "iou": IoULoss,
    "lovasz": LovaszLossMultiLabel,
    "focal" : FocalLossBinary
}

class ComposedLossWithLogits(nn.Module):

    def __init__(self, names_and_weights):
        super().__init__()

        assert type(names_and_weights) in (dict, list)

        if isinstance(names_and_weights, dict):
            names_and_weights = names_and_weights.items()

        self.names = []
        self.loss_fns = []
        weights = []

        for name, weight in names_and_weights:
            if weight == 0:
                continue
            self.names.append(name)
            self.loss_fns.append(str2Loss[name]())
            weights.append(weight)

        self.loss_fns = nn.ModuleList(self.loss_fns)

        self.register_buffer('weights', torch.Tensor(weights))
        self.weights /= self.weights.sum()

    def forward(self, logit, target):
        losses = torch.empty(7)
        for c in range(7):
            c_loss = torch.stack([loss_fn(logit[:, c, :, :], target.float())
                                    for loss_fn in self.loss_fns])
            c_sumed_loss = (torch.Tensor(c_loss) * self.weights).sum()
            losses[c] = c_sumed_loss
        loss = torch.Tensor(losses).mean()


        return loss#sumed_loss, losses


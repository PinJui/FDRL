import torch
from torch import nn


class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=7, feat_dim=128, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(
                self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(
                torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        center = self.centers[labels]
        dist = (x-center).pow(2).sum(dim=-1)
        loss = torch.clamp(dist, min=1e-12, max=1e+12).mean(dim=-1)
        return loss


class CompactnessLoss(nn.Module):
    def __init__(self, num_branch=9, feat_dim=128, use_gpu=True):
        super(CompactnessLoss, self).__init__()
        self.num_branch = num_branch
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(
                self.num_branch, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(
                torch.randn(self.num_branch, self.feat_dim))

    def forward(self, x):
        dist = (x-self.centers).pow(2).sum(dim=-1).sum(dim=-1)
        loss = torch.clamp(dist, min=1e-12, max=1e+12).mean(dim=-1)
        return loss

class BalanceLoss(nn.Module):
    def __init__(self, num_branch):
        super(BalanceLoss, self).__init__()
        self.num_branch = num_branch

    def forward(self, x):
        w_u = torch.zeros_like(x) + (1 / self.num_branch)
        w_bar = x.mean(dim=0)
        return torch.norm(w_bar - w_u, p=1)

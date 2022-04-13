import torch
from torch import nn
from torchvision import models


class FDN(nn.Module):
    def __init__(self, num_branch, basic_dim, feat_dim):
        super(FDN, self).__init__()
        # M denotes the feature dimension, 9 in paper
        self.M = num_branch
        # P denotes the basic feature dimension, 512 in paper
        self.P = basic_dim
        # D denotes the feature dimension, 128 in paper
        self.D = feat_dim
        for i in range(self.M):
            setattr(self, "fdn_fc%d" % i, nn.Sequential(nn.Linear(
                self.P, self.D), nn.ReLU()))

    def forward(self, x):
        features = []
        for i in range(self.M):
            feature = getattr(self, "fdn_fc%d" % i)(x)
            features.append(feature)
        
        features = torch.stack(features).permute([1, 0, 2])
        return features


class IntraRM(nn.Module):
    def __init__(self, num_branch, feat_dim):
        super(IntraRM, self).__init__()
        # M denotes the feature dimension, 9 in paper
        self.M = num_branch
        # D denotes the feature dimension, 128 in paper
        self.D = feat_dim
        for i in range(self.M):
            setattr(self, "intra_fc%d" % i, nn.Sequential(nn.Linear(
                self.D, self.D), nn.Sigmoid()))

    def forward(self, x):
        features = []
        alphas = []
        for i in range(self.M):
            # dim of x: [batch, branch, feat_dim]
            feature = getattr(self, "intra_fc%d" % i)(x[:, i, :])
            alpha = torch.norm(feature, p=1, dim=-1)
            alphas.append(alpha)
            features.append(alpha.unsqueeze(1) * x[:, i, :])

        features = torch.stack(features).permute([1, 0, 2])
        alphas = torch.stack(alphas).permute([1, 0])
        return features, alphas


class InterRM(nn.Module):
    def __init__(self, num_branch, feat_dim):
        super(InterRM, self).__init__()
        # M denotes the feature dimension, 9 in paper
        self.M = num_branch
        # D denotes the feature dimension, 128 in paper
        self.D = feat_dim
        for i in range(self.M):
            setattr(self, "inter_fc%d" % i, nn.Sequential(nn.Linear(
                self.D, self.D), nn.ReLU()))
        self.dummy = 1 - torch.eye(self.M, self.M)
        self.delta = 0.5

    def forward(self, x):
        features = []
        for i in range(self.M):
            feature = getattr(self, "inter_fc%d" % i)(x[:, i, :])
            features.append(feature)

        # contiguous to resolve cdist backward issue
        features = torch.stack(features).permute([1, 0, 2]).contiguous()
        dist_mat = torch.cdist(features, features)
        dist_mat = torch.tanh(dist_mat)
        dist_mat = dist_mat * self.dummy.to(x.device)
        # dim => [batch, M, M] x [batch, M, D] = [batch, M, D]
        features_hat = torch.matmul(dist_mat, features)

        return self.delta * features + (1 - self.delta) * features_hat


class FRN(nn.Module):
    def __init__(self, num_branch, feat_dim):
        super(FRN, self).__init__()
        # M denotes the feature dimension, 9 in paper
        self.M = num_branch
        # D denotes the feature dimension, 128 in paper
        self.D = feat_dim
        
        self.intra_rm = IntraRM(self.M, self.D)
        self.inter_rm = InterRM(self.M, self.D)

    def forward(self, x):
        x, alphas = self.intra_rm(x)
        x = self.inter_rm(x)
        return x, alphas


class FDRL(nn.Module):
    def __init__(self, num_branch, basic_dim, feat_dim, num_class):
        super(FDRL, self).__init__()
        # M denotes the feature dimension, 9 in paper
        self.M = num_branch
        # P denotes the basic feature dimension, 512 in paper
        self.P = basic_dim
        # D denotes the feature dimension, 128 in paper
        self.D = feat_dim
        self.num_class = num_class
        
        self.backbone = models.resnet18()
        self.backbone.load_state_dict(
            torch.load('models/resnet18_msceleb.pth', map_location='cpu')['state_dict'], strict=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.pooling = nn.AvgPool2d(7, stride=1)
        self.fdn = FDN(self.M, self.P, self.D)
        self.frn = FRN(self.M, self.D)
        self.epn = nn.Linear(self.D, self.num_class)

    def forward(self, x):
        basic_feat = self.backbone(x)
        basic_feat = self.pooling(basic_feat).squeeze()
        fdn_feat = self.fdn(basic_feat)
        frn_feat, alphas = self.frn(fdn_feat)
        frn_feat = torch.sum(frn_feat, dim=1)
        pred = self.epn(frn_feat)
        return fdn_feat, alphas, pred

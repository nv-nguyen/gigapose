from torch import nn
import torch
import torch.nn.functional as F
from torch import nn, einsum


def cosine_similarity(a, b, normalize=True):
    if normalize:
        w1 = a.norm(p=2, dim=1, keepdim=True)
        w2 = b.norm(p=2, dim=1, keepdim=True)
        sim_matrix = torch.mm(a, b.t()) / (w1 * w2.t()).clamp(min=1e-8)
    else:
        sim_matrix = torch.mm(a, b.t())
    return sim_matrix


class ScaleLoss(nn.Module):
    def __init__(self, loss="l2", log=False):
        super(ScaleLoss, self).__init__()
        self.loss = loss
        self.log = log

    def forward(self, pred_scale, gt_scale):
        """
        pred_scale: (B,)
        gt_scale: (B,)
        """
        if self.log:
            pred_scale = torch.log(pred_scale.clamp(min=1e-6))
            gt_scale = torch.log(gt_scale)
        if self.loss == "l1":
            loss = F.l1_loss(pred_scale, gt_scale)
        else:
            loss = F.mse_loss(pred_scale, gt_scale)
        assert not torch.isnan(loss)
        return loss


class InplaneLoss(nn.Module):
    def __init__(self, loss="l2", normalize=False):
        super(InplaneLoss, self).__init__()
        self.normalize = normalize
        self.loss = loss
        self.eps = 1e-6

    def forward(self, pred_cos_sin, gt_cos_sin):
        """
        pred_inp_R: (B, 2)
        gt_inp_R: (B, 2)
        """
        if self.normalize:
            pred_cos_sin = F.normalize(pred_cos_sin, dim=1)
            gt_cos_sin = F.normalize(gt_cos_sin, dim=1)
        if self.loss == "geodesic":
            pred_cos = pred_cos_sin[:, 0]
            pred_sin = pred_cos_sin[:, 1]
            gt_cos = gt_cos_sin[:, 0]
            gt_sin = gt_cos_sin[:, 1]
            cos_diff = pred_cos * gt_cos + pred_sin * gt_sin
            cos_diff = torch.clamp(cos_diff, -1 + self.eps, 1 - self.eps)
            loss = torch.acos(cos_diff).mean()
        elif self.loss == "l1":
            loss = F.l1_loss(pred_cos_sin, gt_cos_sin)
        elif self.loss == "l2":
            loss = F.mse_loss(pred_cos_sin, gt_cos_sin)
        else:
            raise NotImplementedError
        assert not torch.isnan(loss)
        return loss


class InfoNCE(nn.Module):
    def __init__(self, tau=0.1):
        super(InfoNCE, self).__init__()
        self.tau = tau

    def forward(self, query_feat, ref_feats, labels):
        # credit: https://github.com/sthalles/SimCLR/blob/master/simclr.py#L26
        """
        Given a pair (query, ref) as a positive pair, and a set of negative features if available, compute the InfoNCE loss
        """
        query_feat = F.normalize(query_feat, dim=1)
        ref_feats = F.normalize(ref_feats, dim=1)
        logits = query_feat @ ref_feats.t()
        logits = logits / self.tau
        loss = F.cross_entropy(logits, labels)
        return loss


if __name__ == "__main__":
    metric = InfoNCE()
    query_feat = torch.randn(10, 128)
    ref_feat = torch.randn(10, 128)
    neg_feat = torch.randn(20, 128)
    loss = metric(query_feat, ref_feat, neg_feat, 0)
    print(loss)

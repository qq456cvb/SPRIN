# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 13:21:38 2021

@author: eliphat
"""
import os
import h5py
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def zm(a):
    zeros = torch.zeros_like(a)
    ones = torch.ones_like(a)
    return torch.stack([torch.cos(a), -torch.sin(a), zeros, torch.sin(a), torch.cos(a), zeros, zeros, zeros, ones], -1).reshape(*a.shape, 3, 3)


def ym(a):
    zeros = torch.zeros_like(a)
    ones = torch.ones_like(a)
    return torch.stack([torch.cos(a), zeros, torch.sin(a), zeros, ones, zeros, -torch.sin(a), zeros, torch.cos(a)], -1).reshape(*a.shape, 3, 3)


def c2s(x):
    """cartesan to spherical coordinates

    Args:
        x (... x 3): 

    Returns:
        ... x 2: 
    """
    return torch.stack([torch.atan2(x[..., 1], x[..., 0]), torch.acos(torch.clamp(x[..., 2] / torch.norm(x, dim=-1), -1., 1.)), x[..., 2]], -1)


def rotm(x):
    return zm(x[..., 0]) @ ym(x[..., 1]) @ zm(x[..., 2])


def rifeat(points_r, points_s):
    """generate rotation invariant features

    Args:
        points_r (B x N x K x 3): 
        points_s (B x N x 1 x 3): 
    """

    # [*, 3] -> [*, 8] with compatible intra-shapes
    if points_r.shape[1] != points_s.shape[1]:
        points_r = points_r.expand(-1, points_s.shape[1], -1, -1)
    
    r_mean = torch.mean(points_r, -2, keepdim=True)
    l1, l2, l3 = r_mean - points_r, points_r - points_s, points_s - r_mean
    l1_norm = torch.norm(l1, 'fro', -1, True)
    l2_norm = torch.norm(l2, 'fro', -1, True)
    l3_norm = torch.norm(l3, 'fro', -1, True).expand_as(l2_norm)
    theta1 = (l1 * l2).sum(-1, keepdim=True) / (l1_norm * l2_norm + 1e-7)
    theta2 = (l2 * l3).sum(-1, keepdim=True) / (l2_norm * l3_norm + 1e-7)
    theta3 = (l3 * l1).sum(-1, keepdim=True) / (l3_norm * l1_norm + 1e-7)
    
    # spherical mapping
    sx = c2s(points_s)
    sx[..., [0, 2]] = sx[..., [2, 0]]
    sx *= -1
    m = rotm(sx) # B x N x 1 x 3 x 3
    h = torch.norm(points_r, dim=-1, keepdim=True)
    r_s2 = points_r / h
    res = torch.einsum('bnxy,bnky->bnkx', m[:, :, 0], r_s2.expand(m.shape[0], m.shape[1], -1, -1))
    txj_inv_xi = torch.acos(torch.clamp(res[..., 2:3], -1., 1.)) / np.pi
    return torch.cat([txj_inv_xi, h, l1_norm, l2_norm, l3_norm, theta1, theta2, theta3], dim=-1)


def conv_kernel(iunit, ounit, *hunits):
    layers = []
    for unit in hunits:
        layers.append(nn.Linear(iunit, unit))
        layers.append(nn.LayerNorm(unit))
        layers.append(nn.ReLU())
        iunit = unit
    layers.append(nn.Linear(iunit, ounit))
    return nn.Sequential(*layers)


class SparseSO3Conv(nn.Module):
    def __init__(self, rank, n_in, n_out, *kernel_interns, layer_norm=True):
        super().__init__()
        self.kernel = conv_kernel(8, rank, *kernel_interns)
        self.outnet = nn.Linear(rank * n_in, n_out)
        self.rank = rank
        self.layer_norm = nn.LayerNorm(n_out) if layer_norm else None

    def do_conv_ranked(self, r_inv_s, feat):
        # [b, n, k, rank], [b, n, k, cin] -> [b, n, cout]
        kern = self.kernel(r_inv_s).reshape(*feat.shape[:-1], self.rank)
        # PointConv-like optimization
        contracted = torch.einsum("bnkr,bnki->bnri", kern, feat).flatten(-2)
        return self.outnet(contracted)

    def forward(self, feat_points, feat, eval_points):
        eval_points_e = torch.unsqueeze(eval_points, -2)
        r_inv_s = rifeat(feat_points, eval_points_e)
        conv = self.do_conv_ranked(r_inv_s, feat)
        if self.layer_norm is not None:
            return self.layer_norm(conv)
        return conv


class GlobalInfoProp(nn.Module):
    def __init__(self, n_in, n_global):
        super().__init__()
        self.linear = nn.Linear(n_in, n_global)

    def forward(self, feat):
        # [b, k, n_in] -> [b, k, n_in + n_global]
        tran = self.linear(feat)
        glob = tran.max(-2, keepdim=True)[0].expand(*feat.shape[:-1], tran.shape[-1])
        return torch.cat([feat, glob], -1)


class Neighbourhood(nn.Module):

    def knn_indices(self, points_target, points_query, k, stride):
        # points: [*, 3] out: [*, k // stride]
        self_dist = torch.cdist(points_query, points_target)
        _, knn_indices = torch.topk(self_dist, k, -1, False, False)
        knn_indices = knn_indices[..., torch.randperm(knn_indices.size(-1))]
        return knn_indices[..., ::stride]  # [*, Q, K']

    def knn_feat(self, feat, knn_indices):
        uns = feat.unsqueeze(-3)
        ex = uns.expand(feat.shape[0], knn_indices.shape[1], feat.shape[-2], feat.shape[-1])
        return torch.gather(ex, -2, knn_indices.expand(*knn_indices.shape[:-1], feat.shape[-1]))

    def forward(self, points, in_feat, k, stride):
        # points [b, n, 3] in_feat [b, n, c_in] out [b, n, k // stride, c_in]
        ind = self.knn_indices(points, points, k, stride).unsqueeze(-1)
        return self.knn_feat(in_feat, ind)

    def query(self, points_target, feat_target, points_query, k, stride):
        # [b, T, 3] Q [b, Q, 3]
        ind = self.knn_indices(points_target, points_query, k, stride).unsqueeze(-1)
        return self.knn_feat(points_target, ind), self.knn_feat(feat_target, ind)


class NeighbourhoodSparseSO3Conv(nn.Module):
    def __init__(self, k, stride, rank, n_in, n_out, *hunits):
        super().__init__()
        self.conv = SparseSO3Conv(rank, n_in, n_out, *hunits)
        self.k = k
        self.stride = stride
        self.neighbourhood = Neighbourhood()

    def forward(self, feat_points, feat, eval_points):
        
        feat_points, eval_points = feat_points[..., :3], eval_points[..., :3]
        nn, nn_feat = self.neighbourhood.query(
            feat_points, feat,
            eval_points,
            self.k, self.stride
        )
        return self.conv(nn, nn_feat, eval_points)


class AdaptiveGrouping(nn.Module):
    def __init__(self, n_points_target, n_feat_in):
        super().__init__()
        self.tr = nn.Linear(n_feat_in, n_points_target)

    def forward(self, points, feat):
        # points: [b, n, 3], feat [b, n, c_in] -> [b, n', 3]
        act = F.softmax(self.tr(feat), -2)
        return torch.einsum('bnz,bnc->bcz', points, act)


class FPSGrouping(nn.Module):
    def __init__(self, n_points_target, *args):
        super().__init__()
        self.n_points_target = n_points_target

    def farthest_point_sample(self, xyz):
        """
        Input:
            xyz: pointcloud data, [B, N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """
        npoint = self.n_points_target
        device = xyz.device
        B, N, C = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        return centroids

    def index_points(self, points, idx):
        """
        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        Return:
            new_points:, indexed points data, [B, S, C]
        """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points

    def forward(self, points, fps_idx=None):
        return self.index_points(points[..., :3], self.farthest_point_sample(points[..., :3]) if fps_idx is None else fps_idx)


class SPRINCls(nn.Module):
    def __init__(self, n_classes, fps_n):
        super().__init__()
        self.conv_11 = NeighbourhoodSparseSO3Conv(64, 2, 32, 1, 32, 80)
        self.glob_11 = GlobalInfoProp(32, 8)
        self.conv_12 = NeighbourhoodSparseSO3Conv(64, 2, 32, 40, 48, 80)
        self.glob_12 = GlobalInfoProp(48, 16)

        self.pool_1 = FPSGrouping(fps_n, 64)
        self.pconv_1 = NeighbourhoodSparseSO3Conv(72, 3, 64, 64, 128, 100)
        
        self.conv_21 = NeighbourhoodSparseSO3Conv(32, 1, 64, 128, 128, 140)
        self.glob_21 = GlobalInfoProp(128, 16)
        self.conv_22 = NeighbourhoodSparseSO3Conv(32, 1, 72, 144, 144, 180)
        self.glob_22 = GlobalInfoProp(144, 16)

        self.pool_2 = FPSGrouping(32, 160)
        self.pconv_2 = NeighbourhoodSparseSO3Conv(32, 1, 128, 160, 256, 240)

        self.conv_31 = SparseSO3Conv(128, 256, 256, 240)
        self.conv_32 = SparseSO3Conv(128, 256, 256, 240)
        self.glob_3 = GlobalInfoProp(256, 128)
        
        self.top = nn.Sequential(nn.Linear(768, 256), nn.BatchNorm1d(256), nn.ReLU(),
                                 nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(),
                                 nn.Linear(64, n_classes))

    def forward(self, x, fps_idxs=None):
        feat = torch.ones_like(x[..., :1])
        
        feat = self.glob_11(F.relu(self.conv_11(x, feat, x)))
        feat = self.glob_12(F.relu(self.conv_12(x, feat, x)))
        x2 = self.pool_1(x, fps_idxs)
        feat = F.relu(self.pconv_1(x, feat, x2))
        
        feat = self.glob_21(F.relu(self.conv_21(x2, feat, x2)))
        feat = self.glob_22(F.relu(self.conv_22(x2, feat, x2)))
        x3 = self.pool_2(x2)
        feat = F.relu(self.pconv_2(x2, feat, x3))

        feat = F.relu(self.conv_31(x3, feat, x3))
        feat = F.relu(self.conv_32(x3, feat, x3))
        feat = self.glob_3(feat)
        
        feat = torch.cat((torch.max(feat, 1)[0], torch.mean(feat, 1)), dim=1)

        return self.top(feat)
    

class SPRINSeg(nn.Module):
    def __init__(self, n_classes, fps_n):
        super().__init__()
        self.conv_11 = NeighbourhoodSparseSO3Conv(64, 2, 32, 16, 32, 80)
        self.glob_11 = GlobalInfoProp(32, 8)
        self.conv_12 = NeighbourhoodSparseSO3Conv(64, 2, 32, 40, 48, 80)
        self.glob_12 = GlobalInfoProp(48, 16)

        self.pool_1 = FPSGrouping(fps_n, 64)
        self.pconv_1 = NeighbourhoodSparseSO3Conv(32, 1, 64, 64, 128, 100)
        
        self.conv_21 = NeighbourhoodSparseSO3Conv(32, 1, 64, 128, 128, 140)
        self.glob_21 = GlobalInfoProp(128, 16)
        self.conv_22 = NeighbourhoodSparseSO3Conv(32, 1, 72, 144, 144, 180)
        self.glob_22 = GlobalInfoProp(144, 16)

        self.pool_2 = FPSGrouping(32, 160)
        self.pconv_2 = NeighbourhoodSparseSO3Conv(32, 1, 128, 160, 256, 240)

        self.conv_31 = SparseSO3Conv(128, 256, 256, 240)
        self.conv_32 = SparseSO3Conv(128, 256, 256, 240)
        self.glob_3 = GlobalInfoProp(256, 32)
        
        self.upconv_4 = NeighbourhoodSparseSO3Conv(16, 1, 32, 288, 128, 128)
        self.conv_41 = NeighbourhoodSparseSO3Conv(32, 1, 64, 128, 128, 128)
        self.glob_41 = GlobalInfoProp(128, 32)
        
        self.upconv_5 = NeighbourhoodSparseSO3Conv(32, 1, 32, 160, 48, 80)
        self.conv_51 = NeighbourhoodSparseSO3Conv(48, 2, 32, 48, 48, 80)
        self.glob_51 = GlobalInfoProp(48, 16)
        self.conv_52 = NeighbourhoodSparseSO3Conv(96, 3, 32, 64, 64, 80)
        self.glob_52 = GlobalInfoProp(64, 16)
        
        self.top = nn.Sequential(nn.Conv1d(80, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                                 nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
                                 nn.Conv1d(256, n_classes, 1))

    def forward(self, x, fps_idxs, cls_idxs):
        feat = cls_idxs[:, None].expand(-1, x.size(1), -1)
        
        feat = self.glob_11(F.relu(self.conv_11(x, feat, x)))
        feat = self.glob_12(F.relu(self.conv_12(x, feat, x)))
        x2 = self.pool_1(x, fps_idxs)
        feat = F.relu(self.pconv_1(x, feat, x2))
        
        feat = self.glob_21(F.relu(self.conv_21(x2, feat, x2)))
        feat = self.glob_22(F.relu(self.conv_22(x2, feat, x2)))
        x3 = self.pool_2(x2)
        feat = F.relu(self.pconv_2(x2, feat, x3))
        
        feat = F.relu(self.conv_31(x3[:, None], feat[:, None].expand(-1, x3.shape[1], -1, -1), x3))
        feat = F.relu(self.conv_32(x3[:, None], feat[:, None].expand(-1, x3.shape[1], -1, -1), x3))
        feat = self.glob_3(feat)
        
        feat = self.upconv_4(x3, feat, x2)
        feat = self.glob_41(F.relu(self.conv_41(x2, feat, x2)))
        
        feat = self.upconv_5(x2, feat, x)
        feat = self.glob_51(F.relu(self.conv_51(x, feat, x)))
        feat = self.glob_52(F.relu(self.conv_52(x, feat, x)))

        return self.top(feat.transpose(-1, -2))


if __name__ == '__main__':
    model = SPRINCls(40, 128)
    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))

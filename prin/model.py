import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F

from s2cnn import s2_equatorial_grid, S2Convolution, so3_equatorial_grid, SO3Convolution, so3_integrate, soft, S2HConvolution
import numpy as np
from s2cnn import so3_rotation

        

class GammaMean(nn.Module):
    def __init__(self, n_feat) -> None:
        super().__init__()
        self.fc = nn.Linear(1, n_feat)

    def forward(self, x):
        return self.fc(x.mean(-1, keepdim=True))


class Model(nn.Module):
    def __init__(self, nclasses, cfg):
        super().__init__()
        self.features = [1, 40, 40, nclasses]
        self.bandwidths = [cfg.bw,] * len(self.features)
        self.linear1 = nn.Linear(nclasses + 16, 50)
        self.linear2 = nn.Linear(50, 50)
        
        sequence = []
        # S2 layer
        grid = s2_equatorial_grid(max_beta=0, n_alpha=2 * self.bandwidths[0], n_beta=1)
        sequence.append(S2HConvolution(self.features[0], self.features[1], self.bandwidths[0], self.bandwidths[1], grid))
        
        # SO3 layers
        for l in range(1, len(self.features) - 1):
            nfeature_in = self.features[l]
            nfeature_out = self.features[l + 1]
            b_in = self.bandwidths[l]
            b_out = self.bandwidths[l + 1]

            sequence.append(nn.BatchNorm3d(nfeature_in, affine=True))
            sequence.append(nn.ReLU())

            grid = so3_equatorial_grid(max_beta=0, max_gamma=0, n_alpha=2 * b_in, n_beta=1, n_gamma=1)
            sequence.append(SO3Convolution(nfeature_in, nfeature_out, b_in, b_out, grid))

        sequence.append(nn.BatchNorm3d(self.features[-1], affine=True))
        sequence.append(nn.ReLU())
        sequence.append(GammaMean(2 * self.bandwidths[-1]))

        self.sequential = nn.Sequential(*sequence)


    def forward(self, x, target_index, cat_onehot):  # pylint: disable=W0221
        # concat after SO3 conv
        # B * C * a * b * c
        x = self.sequential(x)  # [batch, feature, beta, alpha, gamma]
        # return x
        # B * C * N * 1 * 1
        features = F.grid_sample(x, target_index[:, :, None, None, :])
        # B * N * C
        features = features.squeeze(3).squeeze(3).permute([0, 2, 1]).contiguous()

        # B * N * (C + 16)
        prediction = torch.cat([features, cat_onehot[:, None, :].repeat(1, features.size(1), 1)], dim=2)

        # B * N * C
        prediction = F.relu(self.linear1(prediction))
        prediction = self.linear2(prediction)

        prediction = F.log_softmax(prediction, dim=2)
        return features, prediction


class ModelCls(nn.Module):
    def __init__(self, nclasses, cfg):
        super().__init__()

        self.features = [1, 50, 50, 50, 256, 128, nclasses]
        self.bandwidths = [cfg.bw, cfg.bw, cfg.bw, cfg.bw]

        sequence = []

        # S2 layer
        grid = s2_equatorial_grid(max_beta=0, n_alpha=2 * self.bandwidths[0], n_beta=1)
        sequence.append(S2HConvolution(self.features[0], self.features[1], self.bandwidths[0], self.bandwidths[1], grid))
        
        # SO3 layers
        for l in range(1, len(self.features) - 4):
            nfeature_in = self.features[l]
            nfeature_out = self.features[l + 1]
            b_in = self.bandwidths[l]
            b_out = self.bandwidths[l + 1]

            sequence.append(nn.BatchNorm3d(nfeature_in, affine=True))
            sequence.append(nn.ReLU())

            grid = so3_equatorial_grid(max_beta=0, max_gamma=0, n_alpha=2 * b_in, n_beta=1, n_gamma=1)
            sequence.append(SO3Convolution(nfeature_in, nfeature_out, b_in, b_out, grid))

        sequence.append(nn.BatchNorm3d(self.features[3], affine=True))
        sequence.append(nn.ReLU())

        self.sequential = nn.Sequential(*sequence)

        fcs = []
        # Output layer
        for i in range(3, len(self.features) - 1):
            ch_in = self.features[i]
            ch_out = self.features[i + 1]
            fcs.append(nn.Linear(ch_in, ch_out))
            if i < len(self.features) - 2:
                fcs.append(nn.BatchNorm1d(ch_out))
                fcs.append(nn.ReLU())
        self.out_layer = nn.Sequential(*fcs)

    def forward(self, x):  # pylint: disable=W0221
        x = self.sequential(x)  # [batch, feature, beta, alpha, gamma]
        x = x.mean(-1) # [batch, feature, beta, alpha]
        x = x.view(x.size(0), x.size(1), -1).max(-1)[0]  # [batch, feature]

        x = self.out_layer(x)
        return F.log_softmax(x, dim=1)


def s2_rotation(x, a, b, c):
    x = so3_rotation(x.view(*x.size(), 1).expand(*x.size(), x.size(-1)), a, b, c)
    return x[..., 0]


if __name__ == '__main__':
    model = Model(50).cuda()
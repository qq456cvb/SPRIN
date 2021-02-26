import torch
import numpy as np
import os
import h5py
from scipy.stats import special_ortho_group
from glob import glob
import re

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
class_names = list(seg_classes.keys())
seg_label_delta = {}
seg_label_to_cat = {}
for cat, labels in seg_classes.items():
    for label in labels:
        seg_label_delta[label] = min(labels)
        seg_label_to_cat[label] = cat


def read_data(dir_name, path):
    def read(filepath):
        with h5py.File(filepath, 'r') as fi:
            pcs = fi['data'][:]
            pcs -= pcs.mean(-2, keepdims=True)
            pcs /= np.max(np.linalg.norm(pcs, axis=-1, keepdims=True), -2, keepdims=True)
            pid = np.array(fi['pid'][:])
            for i, c in enumerate(list(pid)):
                pid[i] -= seg_label_delta[c[0]]
            return pcs, pid, fi['pid'][:]
    Q = []
    for f in glob(os.path.join(dir_name, '*')):
        if re.search(path, f):
            Q.append(read(f))
    pcds, pids, origin_pids = [], [], []
    for pcd, pid, lab in Q:
        pcds.append(pcd), pids.append(pid), origin_pids.append(lab)
    return np.concatenate(pcds), np.concatenate(pids), np.concatenate(origin_pids)
    

def fps(xyz, npoint):
    N = xyz.shape[0]
    centroids = np.zeros((npoint,), dtype=np.int64)
    distance = np.ones(N,) * 1e10
    farthest = np.random.randint(N)
    
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :][None]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return centroids


def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data


class ShapeNetPartDataset(torch.utils.data.Dataset):
    def __init__(self, data, pids, origin_pids, ds, rand_rot, aug=False) -> None:
        super().__init__()
        self.data = data
        self.pids = pids
        self.origin_pids = origin_pids
        self.ds = ds
        self.rand_rot = rand_rot
        self.aug = aug
        self.valid_names = list(seg_classes.keys())
        self.valid_idxs = [i for i in range(len(self.data)) if seg_label_to_cat[self.origin_pids[i][0]] in self.valid_names]
        
    def __getitem__(self, idx):
        idx = self.valid_idxs[idx]
        data = self.data[idx]
        pids = self.pids[idx]
        origin_pids = self.origin_pids[idx]

        if self.aug:
            data = jitter_point_cloud(data[None])[0]
        
        npoints = data.shape[0]
        if self.ds is not None:
            idx = np.arange(npoints)
            np.random.shuffle(idx)
            idx = idx[:self.ds]
            if idx.size < self.ds:
                idx = np.concatenate([idx, np.random.randint(npoints, size=(self.ds - idx.size,))])
            data = data[idx]
            pids = pids[idx]
            origin_pids = origin_pids[idx]
            
        if self.rand_rot:
            data = data @ special_ortho_group.rvs(3)
        return data, pids, origin_pids, fps(data, 128), class_names.index(seg_label_to_cat[origin_pids[0]])
            
    def balanced_indices_sample(self):
        ind = np.ones([50,], np.bool)
        for name in self.valid_names:
            ind[seg_classes[name]] = False
        results = []
        while not np.all(ind):
            idx = np.random.randint(len(self))
            valid_idx = self.valid_idxs[idx]
            if np.any(ind[seg_classes[seg_label_to_cat[self.origin_pids[valid_idx][0]]]]):
                continue

            ind[seg_classes[seg_label_to_cat[self.origin_pids[valid_idx][0]]]] = True
            results.append(idx)
        return results

    def __len__(self):
        return len(self.valid_idxs)


class BalancedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.dataset = dataset

    def __iter__(self):
        for i in range((len(self) - 1) // 16 + 1):
            indices = self.dataset.balanced_indices_sample()
            for idx in indices:
                yield idx

    def __len__(self):
        return len(self.dataset)

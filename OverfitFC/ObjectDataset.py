import os
import pickle

import numpy as np
import torch
import trimesh as tm
from torch import nn
from torch.utils.data import DataLoader, Dataset


class ObjectDataset(Dataset):
    def __init__(self, shapenet_base_path, sample_name_path, sample_pointcloud_path):
        self.base_path = shapenet_base_path
        self.cats_and_names = pickle.load(open(sample_name_path, 'rb'))
        self.pointclouds = torch.load(sample_pointcloud_path)

    def __len__(self):
        return len(self.pointclouds)
    
    def __getitem__(self, idx):
        rand_idx = torch.randint(0, 10000, [1000], device='cuda', dtype=torch.long)
        return self.pointclouds[idx][rand_idx, :], idx


if __name__ == '__main__':
    dataset = ObjectDataset('C:\\Users\\24jas\\Desktop\\TouchFilter\\ForceClosure\\data\\Reconstructions\\2000')
    print(len(dataset))
    print(dataset.obj_codes.shape)
    print(dataset.obj_pointclouds.shape)
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    for a, b, c in dataloader:
        print(a.shape, b.shape, c)

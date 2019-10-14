from torch.utils.data import Dataset
import os
import numpy as np
import networkx as nx
import torch


class GraphDataset(Dataset):
    """
    Graph FFMPEG Dataset
    LABELS: 1 = CLANG, 0 = GCC
    """
    def __init__(self, path='data/'):
        self.path = path
        self.files = os.listdir(self.path)
        self.pad_size = 64

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        graph = nx.read_gpickle(self.path + self.files[idx])
        label = 1 if '-clang-' in self.files[idx] else 0
        features = torch.tensor(np.array([graph.nodes[node]['features'] for node in graph.nodes]))
        adj = torch.tensor(nx.adjacency_matrix(graph).todense().astype('float32'))
        f = torch.zeros(self.pad_size, features.shape[1])
        a = torch.zeros(self.pad_size, self.pad_size)
        real_size = min(self.pad_size, features.shape[0])
        a[:real_size, :real_size] = adj[:real_size, :real_size]
        f[:real_size] = features[:real_size]
        l = torch.zeros(1)
        l[:] = label
        return (f, a), l

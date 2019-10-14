from torch.utils.data import Dataset
import os
import networkx as nx


class GraphDataset(Dataset):
    """
    Graph FFMPEG Dataset
    LABELS: 1 = CLANG, 0 - GCC
    """
    def __init__(self, path='data/'):
        self.path = path
        self.files = os.listdir(self.path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        graph = nx.read_gpickle(self.path + self.files[idx])
        label = 1 if '-clang-' in self.files[idx] else 0
        return graph
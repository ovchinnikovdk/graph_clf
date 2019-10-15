import torch
from torch.nn.parameter import Parameter
import torch.sparse as tsparse


class GraphConv(torch.nn.Module):
    """
    Inspired by: https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Init params
        self.w = Parameter(torch.FloatTensor(in_features, out_features))
        stdv = 1. / (self.w.size(1) ** 0.5)
        self.w.data.uniform_(-stdv, stdv)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, x, adj):
        support = torch.matmul(x, self.w)
        output = torch.matmul(adj, support)
        return output if self.bias is None else output + self.bias


class GCN(torch.nn.Module):
    def __init__(self, in_features=702, out_features=1, n_hidden=128):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_hidden = n_hidden
        self.pad_size = 128

        self.gconv1 = GraphConv(in_features, n_hidden)
        self.relu = torch.nn.ReLU()
        self.gconv2 = GraphConv(n_hidden, n_hidden)
        self.fc = torch.nn.Sequential(torch.nn.Linear(self.pad_size * n_hidden, n_hidden),
                                      torch.nn.Dropout(0.4),
                                      torch.nn.Linear(n_hidden, out_features),
                                      torch.nn.Sigmoid())

    def forward(self, input):
        x, adj = input
        x = self.gconv1(x, adj)
        x = self.relu(x)
        x = self.gconv2(x, adj)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

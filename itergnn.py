import dgl
import torch
import torch.nn as nn

from IterGNN.core import IterativeModule
from IterGNN.experiments.VI.generate_graph import generate_graph
from IterGNN.nn.MLP import MLP
from IterGNN.nn.MPNN import MPNNLayer
from IterGNN.nn.PathGNN import PathGNNLayer
from IterGNN.utils.gnn_utils import BasicReadout


class gmodule(nn.Module):

    def __init__(self):
        super(gmodule, self).__init__()
        self.gnn = MPNNLayer(1, 1, 1, 1)
        self.mlp = MLP(1, 1, out_act='Sigmoid')
        self.readtout = BasicReadout('sum')

    def forward(self, g, nf, ef):
        with g.local_scope():
            unf, _ = self.gnn(g, nf, ef)
            rd = self.readtout(g, self.mlp(unf), None)
            return torch.sigmoid(rd)


def main():
    f_module = PathGNNLayer(1)
    g_module = gmodule()
    iter_module = IterativeModule(f_module, g_module)
    gs = dgl.batch([generate_graph(5, 10), generate_graph(3, 5)])
    nf, ef, info = iter_module(gs, gs.ndata['feat'], gs.edata['feat'])
    print(nf, ef)


if __name__ == '__main__':
    main()

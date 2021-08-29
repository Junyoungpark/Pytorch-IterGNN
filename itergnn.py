import dgl
from dgl.nn.pytorch.utils import Sequential

from IterGNN.core import IterativeModule
from IterGNN.experiments.VI.generate_graph import generate_graph
from IterGNN.nn.MPNN import MPNNLayer
from IterGNN.utils.gnn_utils import BasicReadout


def main():
    f_module = MPNNLayer(1, 1, 1, 1)
    g_module = Sequential(MPNNLayer(1, 1, 1, 1),
                          BasicReadout('sum'))

    iter_module = IterativeModule(f_module, g_module)
    gs = dgl.batch([generate_graph(5, 10), generate_graph(3, 5)])
    nf, ef, info = iter_module(gs, gs.ndata['feat'], gs.edata['feat'])


if __name__ == '__main__':
    main()

import dgl
import torch
from torch import nn


class IterativeModule(nn.Module):

    def __init__(self,
                 iteration_body: nn.Module,  # f(.)
                 condition_module: nn.Module,  # g(.)
                 stop_epsilon: float,
                 decay_lambda: float):
        super(IterativeModule, self).__init__()

        self.iteration_body = iteration_body
        self.condition_module = condition_module
        self.register_buffer('eps', torch.ones(1) * stop_epsilon)
        self.register_buffer('decay_lambda', torch.ones(1) * decay_lambda)  # Decaying param; Algo. 3 in Appendix

    def forward(self, graph: dgl.graph, node_feat: torch.tensor, edge_feat: torch.tensor):
        k = 1  # overall iteration count
        decay_lambda = self.decay_lambda

        cum_cs = torch.ones(size=(graph.batch_size(),), device=node_feat.device)

        stops = torch.zeros(size=(graph.batch_size(),)).bool()  # indicators
        stop_ks = torch.ones(size=(graph.batch_size(),), device=node_feat.device)  # integer valued

        nf, ef = node_feat, edge_feat
        nfs, efs, css = [], [], []

        while True:
            nf, ef = self.iteration_body(graph, nf, ef)  # [#. total nodes x nf_dim], [#. total edges x ef_dim]
            cs = self.condition_module(graph, nf, ef)  # [#. graphs x 1]
            cum_cs *= (1 - cs)

            stop, stop_ks = self._get_stop(stop_ks, cum_cs, decay_lambda, k)

            if stop:
                break
            else:
                k += 1
                decay_lambda *= decay_lambda
                nfs.append(nf), efs.append(ef), css.append(cs)

        # compute the weighted hidden node and edge embedding
        
    def _get_stop(self,
                  stop_ks: torch.tensor,
                  cum_cs: torch.tensor,
                  decay_lambda: torch.tensor,
                  cur_k: int):
        to_stop = (decay_lambda * cum_cs > self.eps).view(-1)
        stop_ks[to_stop] = torch.min(stop_ks[to_stop], torch.ones_like(stop_ks[to_stop]) * cur_k)
        return to_stop.all(), stop_ks

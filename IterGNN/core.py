import dgl
import torch
from torch import nn


class IterativeModule(nn.Module):

    def __init__(self,
                 iteration_body: nn.Module,  # f(.)
                 condition_module: nn.Module,  # g(.)
                 stop_epsilon: float = 0.001,
                 decay_lambda: float = 0.999):
        super(IterativeModule, self).__init__()

        self.iteration_body = iteration_body
        self.condition_module = condition_module
        self.register_buffer('eps', torch.ones(1) * stop_epsilon)
        self.register_buffer('decay_lambda', torch.ones(1) * decay_lambda)  # Decaying param; Algo. 3 in Appendix

    def forward(self, graph: dgl.graph, node_feat: torch.tensor, edge_feat: torch.tensor):
        ## asserting at least one message passing happens

        k = 1  # overall iteration count
        decay_lambda = self.decay_lambda
        cum_cs = torch.ones(size=(graph.batch_size, 1), device=node_feat.device)
        stop_ks = torch.ones(size=(graph.batch_size,), device=node_feat.device) * float('inf')  # integer valued

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
        stop_ks = stop_ks.long() - 1  # to be used as indices

        nfs = torch.stack(nfs, dim=1)  # [#. nodes x max. iter len x nf dim]
        efs = torch.stack(efs, dim=1)  # [#. edges x max. iter len x ef dim]
        css = torch.stack(css, dim=1)  # [#. graphs x max. iter len x 1]

        mask = torch.zeros(graph.batch_size, nfs.shape[1], 1)  # [# .graphs x max. iter len x 1]
        for i, (cs, stop_k) in enumerate(zip(css, stop_ks)):
            mask[i] = (cs * torch.cat([torch.ones(1, 1), (1 - cs).cumprod(dim=0)[1:]], dim=0))
            mask[i, stop_k:, 0] = 0.0
        nfs = (nfs * mask.repeat_interleave(graph.batch_num_nodes(), dim=0)).sum(dim=1)  # sum over iter index
        efs = (efs * mask.repeat_interleave(graph.batch_num_edges(), dim=0)).sum(dim=1)  # sum over iter index

        info = {
            'mask': mask,
            'css': css,
            'stop_ks': stop_ks
        }

        return nfs, efs, info

    def _get_stop(self,
                  stop_ks: torch.tensor,
                  cum_cs: torch.tensor,
                  decay_lambda: torch.tensor,
                  cur_k: int):
        to_stop = (decay_lambda * cum_cs <= self.eps).view(-1)
        stop_ks[to_stop] = torch.min(stop_ks[to_stop], torch.ones_like(stop_ks[to_stop]) * cur_k)
        return to_stop.all(), stop_ks

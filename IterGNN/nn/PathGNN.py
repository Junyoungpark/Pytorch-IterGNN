import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn

from IterGNN.nn.MLP import MLP


class PathGNNLayer(nn.Module):
    def __init__(self,
                 dim: int,
                 **mlp_params):
        super(PathGNNLayer, self).__init__()
        self.edge_model = MLP(input_dim=3 * dim,
                              output_dim=dim,
                              **mlp_params)
        self.attn_model = MLP(input_dim=3 * dim,
                              output_dim=1,
                              **mlp_params)

    def forward(self,
                g: dgl.DGLGraph,
                nf: torch.Tensor,
                ef: torch.Tensor):
        with g.local_scope():
            g.ndata['h'] = nf
            g.edata['h'] = ef

            # perform edge update
            g.apply_edges(func=self.edge_update)

            # compute attention score
            g.edata['attn'] = dglnn.edge_softmax(g, self.attn_model(g.edata['em_input']))

            # update nodes
            g.update_all(message_func=self.message_func,
                         reduce_func=dgl.function.sum('m', 'agg_m'),
                         apply_node_func=self.node_update)

            updated_ef = g.edata['uh']
            updated_nf = g.ndata['uh']
            return updated_nf, updated_ef

    def edge_update(self, edges):
        sender_nf = edges.src['h']
        receiver_nf = edges.dst['h']
        ef = edges.data['h']
        em_input = torch.cat([ef, sender_nf, receiver_nf], dim=-1)
        updated_ef = self.edge_model(em_input)
        return {'uh': updated_ef, 'em_input': em_input}

    @staticmethod
    def message_func(edges):
        return {'m': edges.data['uh'] * edges.data['attn']}

    def node_update(self, nodes):
        agg_m = nodes.data['agg_m']
        nf = nodes.data['h']
        uh = torch.max(nf, agg_m)
        return {'uh': uh}

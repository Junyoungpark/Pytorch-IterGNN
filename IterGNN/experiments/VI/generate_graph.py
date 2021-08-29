from functools import partial

import dgl
import numpy as np
import torch


def generate_graph(nA: int, nS: int, reward_bnd=[-1, 1], gamma=0.9, n_iters=200, atol=1e-3):
    assert nS - 1 > nA, "(nS-1) must be greater than nA"

    u, v = [], []
    for i in range(nS):
        srcs = list(range(nS))
        _ = srcs.pop(i)  # ignore self edges
        src = list(np.random.choice(srcs, nA))

        dst = [i] * nA

        u.extend(src)
        v.extend(dst)

    g = dgl.graph((torch.tensor(u).long(),
                   torch.tensor(v).long()),
                  num_nodes=nS)

    r = np.random.uniform(low=reward_bnd[0], high=reward_bnd[1], size=(g.number_of_edges(), 1))
    g.edata['r'] = torch.tensor(r).float()

    g, converge_step = value_iteration(g, gamma, n_iters, atol=atol)

    # setup features
    g.ndata['feat'] = torch.ones(g.number_of_nodes(), 1)  # dummy feature
    g.edata['feat'] = g.edata['r']
    g.ndata['target'] = g.ndata['value'].clone()

    return g


def compute_action_value(edges, gamma, value_key):
    r = edges.data['r']
    value = edges.src[value_key]
    return {'action_value': r + gamma * value}


def get_max(nodes, value_key, policy_key):
    value, index = nodes.mailbox['action_value'].max(dim=1)
    return {value_key: value, policy_key: index}


def vi_backup(g, gamma, value_key='value', policy_key='policy'):
    g.pull(g.nodes(),
           message_func=partial(compute_action_value, gamma=gamma, value_key=value_key),
           reduce_func=partial(get_max, value_key=value_key, policy_key=policy_key))


def value_iteration(g, gamma=0.9, n_iters=200, value_init=None, atol=1e-3):
    if value_init is None:
        g.ndata['value'] = torch.zeros(g.number_of_nodes(), 1)
    else:
        g.ndata['value'] = value_init

    converge_step = -1
    for i in range(n_iters):
        val_prev = g.ndata['value']
        vi_backup(g, gamma=gamma)
        val_next = g.ndata['value']

        if torch.allclose(val_next, val_prev, atol=atol):
            converge_step = i
            break

    assert converge_step >= 0, "VI doesn't converge."
    return g, converge_step


def get_policy(g, value, gamma=0.9):
    with g.local_scope():
        g.ndata['_value'] = value
        vi_backup(g, gamma, value_key='_value', policy_key='_policy')
        return g.ndata.pop('_policy')

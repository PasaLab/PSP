import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from collections import OrderedDict
import torchvision.transforms as transforms
from torch.nn import Module
from torch_geometric.nn.conv import *
from torch.nn import Sequential, ReLU, Linear

from macro_space.base_layer import GeoLayer
from geniepath import GeniePathLayer
from arma import ARMAConv_

GNN_LIST = [
    "identity",         # indicates: zero
    "appnp",
    "gcn",
    "gat_1",
    "gat_4",
    "gat_8",
    "sage_mean",
    "sage_max",
    "sage_sum",
    "arma",
    "cheb",
    
    'gin',

    'gat_sym',
    'cos',
    'linear',  # gat_linear
    'generalized_linear',
]

ACT_LIST = [
    "tanh", "relu", "linear", "elu", "leaky_relu"
]

FEATURE_ENGINE_LIST = [
    "origin"
    # "svd", "edge_weights", "normalize", "degere_bins"
]


class LinearConv(Module):
    def __init__(self,
                in_channels,
                out_channels,
                bias=True):
        super(LinearConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = torch.nn.Linear(in_channels, out_channels, bias)

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        return self.linear(x)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class ZeroConv(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True):
        super(ZeroConv, self).__init__()
        self.out_dim = out_channels

    def reset_parameters(self):
        pass

    def forward(self, x, edge_index, edge_weight=None):
        return torch.zeros([x.size(0), self.out_dim]).to(x.device)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class Identity(Module):
    def __init__(self,
                 in_dim,
                 out_dim):
        super(Identity, self).__init__()
        assert in_dim == out_dim
        self.dim = in_dim

    def reset_parameters(self):
        pass

    def forward(self, x, edge_index, edge_weight=None):
        return x


def act_map(act):
    if act == "linear":
        return lambda x: x
    elif act == "elu":
        return F.elu
    elif act == "sigmoid":
        return torch.sigmoid
    elif act == "tanh":
        return torch.tanh
    elif act == "relu":
        return torch.nn.functional.relu
    elif act == "relu6":
        return torch.nn.functional.relu6
    elif act == "softplus":
        return torch.nn.functional.softplus
    elif act == "leaky_relu":
        return torch.nn.functional.leaky_relu
    else:
        raise Exception("wrong activate function")


def gnn_map(gnn_name, in_dim, out_dim, num_feat, dropout, concat=True, bias=True):
    '''
    :param gnn_name:
    :param in_dim:
    :param out_dim:
    :param concat: for gat, concat multi-head output or not
    :return: GNN model
    '''
    
    if gnn_name == "gat_8":
        return GATConv(in_dim, out_dim, 8, concat=False, bias=bias, dropout=dropout)
    elif gnn_name == "gat_4":
        return GATConv(in_dim, out_dim, 4, concat=False, bias=bias, dropout=dropout)
    elif gnn_name == "gat_2":
        return GATConv(in_dim, out_dim, 2, concat=False, bias=bias, dropout=dropout)
    elif gnn_name in ["gat_1", "gat"]:
        return GATConv(in_dim, out_dim, 1, concat=False, bias=bias, dropout=dropout)
    elif gnn_name == "gcn":
        return GCNConv(in_dim, out_dim)
    elif gnn_name == "cheb":
        return ChebConv(in_dim, out_dim, K=2, bias=bias)
    elif gnn_name == "sage_mean":
        return SAGEConv(in_dim, out_dim, bias=bias, aggr='mean')
    elif gnn_name == "sage_max":
        return SAGEConv(in_dim, out_dim, bias=bias, aggr='max')
    elif gnn_name == "sage_sum":
        return SAGEConv(in_dim, out_dim, bias=bias, aggr='add')
    elif gnn_name == "gated":
        return GatedGraphConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "arma":
        return ARMAConv_(in_dim, out_dim, bias=bias)
    elif gnn_name == "sg":
        return SGConv(in_dim, out_dim, bias=bias)

    elif gnn_name == "zero":
        return ZeroConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "appnp":
        return APPNP(K=10, alpha=0.1)

    elif gnn_name == 'gin':
        nn1 = Sequential(Linear(in_dim, out_dim), ReLU(), Linear(out_dim, out_dim))
        return GINConv(nn1)

    elif gnn_name in ['gat_sym', 'cos', 'linear', 'generalized_linear']:
        head_num = 8
        return GeoLayer(in_dim, int(out_dim / head_num), heads=head_num, att_type=gnn_name, dropout=dropout, concat=concat)

    elif gnn_name == 'geniepath':
        return GeniePathLayer(in_dim, out_dim)
    
    elif gnn_name == 'gat_relation':
        g = nn.Linear(in_dim * 2, out_dim)
        return g

    elif gnn_name == 'identity':
        return Identity(in_dim, out_dim)

    else:
        raise Exception("wrong gnn name")


class MicroSearchSpace(object):
    def __init__(self, search_space=None, max_cell=10):
        if search_space:
            self.search_space = search_space
        else:
            self.search_space = {}
            self.search_space["gnn"] = GNN_LIST  # gnn type
            self.search_space["act"] = ACT_LIST  # activate function
            self.search_space["concat_type"] = ["add", "product", "concat", "max"]  # same as self_index,
            self.search_space['learning_rate'] = [1e-2, 1e-3, 1e-4, 5e-3, 5e-4]
            self.search_space['dropout'] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            self.search_space['weight_decay_gnns'] = [1e-2, 1e-3, 1e-4, 1e-5, 5e-3, 5e-4, 5e-5]
            self.search_space['weight_decay_fcs'] = [1e-2, 1e-3, 5e-2, 5e-3, 5e-4, 5e-5, 5e-6]
            self.search_space['hidden_unit'] = [32, 64, 128, 256, 512]
            self.search_space['feature_engine'] = FEATURE_ENGINE_LIST

    def get_search_space(self, num_of_nodes):
        actual_actions = []
        for i in range(num_of_nodes):
            prev_index_list = {
                'name': f"prev_{i}",
                'value': list(range(2 + i))
            }
            actual_actions.append(prev_index_list)

            cur_aggregator = {
                    'name': f"gnn_{i}",
                    'value': self.search_space["gnn"]
                }
            actual_actions.append(cur_aggregator)

            activate_func = {
                'name': f"activate_{i}",
                'value': self.search_space["act"]
            }
            actual_actions.append(activate_func)

        flag = False
        for key, value in self.search_space.items():
            if key == 'concat_type':
                flag = True
            if flag:
                cur_op = {
                    'name': key,
                    'value': value
                }
                actual_actions.append(cur_op)
        return actual_actions


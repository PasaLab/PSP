import os
import copy
import numpy as np
import logging
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from collections import OrderedDict
import torchvision.transforms as transforms
from torch.nn import Module
from torch_geometric.nn.conv import *

from constant import *


def demical_to_bin(x):
    _x = x
    res = []
    while _x > 0:
        res.append(_x % 2)
        _x = int(_x / 2)
    while len(res) < len(OPERATIONS_BASE[-1]["value"]):
        res.append(0)
    res.reverse()
    return res


class MacroUtils(object):
    def __init__(self, OPERATIONS):
        self.OPERATIONS = OPERATIONS
        self.REMAIN_FEA_LOWER_BOUND = [
            int(math.ceil(len(OPERATIONS[i]["value"]) / 2)) for i in range(len(self.OPERATIONS))
        ]
        # self.REMAIN_FEA_LOWER_BOUND.append(int(FEATURE_ENGINE_NUM / 2))
        # self.NONEED_PRUNED_FEATURE_ID = [0, 1, 14, 15, 16] if len(self.OPERATIONS[1]["value"]) == 12 else [0, 1, 10, 11, 12]
        # self.NONEED_PRUNED_FEATURE_ID = [0, 1, 14, 15, 16, 29, 30, 31, 32, 45, 46, 47, 48, 49] if len(self.OPERATIONS[1]["value"]) == 12 else [0, 1, 10, 11, 12, 21, 22, 23, 24]
        # get_logger().info(f"NO NEED PRUNED FEATURE ID: {self.NONEED_PRUNED_FEATURE_ID}")
        self.NONEED_PRUNED_FEATURE_ID = []
        self.PRUNE_NUM_FIRST = PRUNE_NUM_FIRST
        self.PRUNE_NUM_SECOND = PRUNE_NUM_SECOND
        self.PRUNE_NUM_LARGER = PRUNE_NUM_LARGER
        self.PRUNE_NUM = self.PRUNE_NUM_FIRST

        self.select_prob = copy.deepcopy(self.OPERATIONS)

    def generate_arch(self, n):
        def _get_arch():
            arch = []
            for op in self.OPERATIONS:
                cur_op_num = len(op["value"]) if op["name"] != "feature_engine" else FEATURE_ENGINE_COMBINE_NUM
                # cur_op = np.random.randint(0, cur_op_num)
                cur_op = random.randint(0, cur_op_num - 1)
                if op["name"] == "feature_engine":
                    while cur_op == 0:
                        # cur_op = np.random.randint(0, cur_op_num)
                        cur_op = random.randint(0, cur_op_num - 1)

                arch.append(cur_op)
            return arch
        archs = []
        while len(archs) < n:
            arch = _get_arch()
            if arch not in archs:
                archs.append(arch)
        return archs

    def convert_to_features(self, arch):
        res = []
        for i in range(len(arch)):
            op = arch[i]
            if i < len(arch) - 1:
                op_list = self.OPERATIONS[i]["value"]
                tmp = [0 for _ in range(len(op_list))]
                tmp[op] = 1
            else:
                # process feature engine
                tmp = demical_to_bin(op)
            res += tmp
        return res

    def parse_feature_to_cond(self, feature_name, arch):
        feature_name = feature_name.split('_')
        op_i = int(feature_name[1])
        op_choice = int(feature_name[3])

        pos = 0
        for i in range(op_i):
            pos += len(self.OPERATIONS[i]["value"])
        # if self.OPERATIONS[op_i]["name"] == "feature_engine":
            # mutil_hot_feature = demical_to_bin(op_choice)
            # is_feature_match_arc = True
            # for i in range(len(mutil_hot_feature)):
            #     if (mutil_hot_feature[i] == 1 and arch[pos+i] == 1) or (mutil_hot_feature[i] == 0 and arch[pos+i] == 0):
            #         continue
            #     is_feature_match_arc = False
            #     break
            # return is_feature_match_arc
        # else:
        #     pos += op_choice
        #     return arch[pos] == 1
        pos += op_choice
        return arch[pos] == 1

    def generate_constrained_arch(self, n, pruned_operations, *args):
        def _get_arch_hard():
            arch = []
            fea_base = 0
            for i in range(len(self.OPERATIONS)):
                candidates = []
                fea_base += len(self.OPERATIONS[i - 1]["value"]) if i > 0 else 0
                # get_logger().info(fea_base)
                if self.OPERATIONS[i]["name"] == "feature_engine":
                    for j in range(FEATURE_ENGINE_COMBINE_NUM):
                        mutil_hot_feature = demical_to_bin(j)
                        is_j_ok = True
                        for k in range(len(mutil_hot_feature)):
                            if mutil_hot_feature[k] == 0:
                                continue
                            fea = fea_base + k
                            if fea in pruned_operations and pruned_operations[fea] is True:
                                is_j_ok = False
                                break
                        if is_j_ok:
                            candidates.append(j)
                else:
                    for j in range(len(self.OPERATIONS[i]["value"])):
                        fea = fea_base + j
                        if fea in pruned_operations and pruned_operations[fea] is True:
                            continue
                        candidates.append(j)
                # get_logger().info(f"op:{i} candidates:{candidates}")

                # gen_op = np.random.choice(candidates)
                gen_op = random.choice(candidates)
                if self.OPERATIONS[i]["name"] == "feature_engine":
                    while gen_op == 0:
                        # gen_op = np.random.choice(candidates)
                        gen_op = random.choice(candidates)
                arch.append(gen_op)
            return arch

        def _get_arch_soft():
            arch = []
            for i in range(len(self.OPERATIONS)):
                if self.OPERATIONS[i]['name'] != 'feature_engine':
                    # get_logger().info(f"all_num: {len(self.OPERATIONS[i]['value'])}, prob: {self.select_prob[i]['value']}")
                    gen_op = np.random.choice(len(self.OPERATIONS[i]['value']), 1, p=self.select_prob[i]['value'])
                    gen_op = gen_op[0]
                else:
                    fe_select_num = max(np.random.choice(len(self.OPERATIONS[i]['value'])), 1)
                    gen_op = np.random.choice(len(self.OPERATIONS[i]['value']), fe_select_num, replace=False, p=self.select_prob[i]['value'])
                    gen_op = sorted(gen_op)
                    x = 0
                    for j in range(len(self.OPERATIONS[i]['value'])):
                        x = x << 1
                        if j in gen_op:
                            x += 1
                    gen_op = x
                arch.append(gen_op)
            return arch

        archs = []
        while len(archs) < n:
            arch = _get_arch_soft() if args[0] == 'soft' else _get_arch_hard()
            if arch not in archs:
                archs.append(arch)
        # get_logger().info(f"gen constrained arcs:\n")
        # for arch in archs:
        #     get_logger().info(arch)
        return archs

    def transform_to_valid_value(self, arch):
        structure = []
        for i in range(len(arch) - 1):
            structure.append(self.OPERATIONS[i]["value"][arch[i]])
        mutil_hot_feature = demical_to_bin(arch[-1])
        feature_engine_selected = []
        for i in range(len(mutil_hot_feature)):
            if mutil_hot_feature[i] == 1:
                feature_engine_selected.append(self.OPERATIONS[-1]["value"][i])
        structure.append(feature_engine_selected)
        return structure

    def valid_fea_num_trans_to_str(self, val):
        tmp = 0
        for i in range(len(self.OPERATIONS)):
            tmp += len(self.OPERATIONS[i]["value"])
            if (val + 1) > tmp:
                continue
            tmp -= len(self.OPERATIONS[i]["value"])
            rest = val - tmp
            return str(self.OPERATIONS[i]["value"][rest])
        # rest = val - 65
        # mutil_hot_feature = demical_to_bin(rest)
        # feature_engine_name_list = ""
        # for i in range(len(mutil_hot_feature)):
        #     if mutil_hot_feature[i] == 1:
        #         feature_engine_name_list = feature_engine_name_list + str(self.OPERATIONS[-1]["value"][i]) + "#"
        # return feature_engine_name_list

    def valid_fea_num_trans_to_i(self, val):
        tmp = 0
        for i in range(len(self.OPERATIONS)):
            tmp += len(self.OPERATIONS[i]["value"])
            if (val + 1) > tmp:
                continue
            tmp -= len(self.OPERATIONS[i]["value"])
            return i
        return len(self.OPERATIONS) - 1

    def combined_fea_trans_to_valid(self, fea_str):
        fea_id = fea_str.split(',')

        feature_name_valid = ""
        for i in range(len(fea_id)):
            op_id = int(fea_id[i])
            op_i = self.valid_fea_num_trans_to_i(op_id)
            op_choice = self.valid_fea_num_trans_to_str(op_id)

            feature_name_valid = feature_name_valid + self.OPERATIONS[op_i]["name"] + " is " + op_choice

            if i < len(fea_id) - 1:
                feature_name_valid = feature_name_valid + ", "

        return feature_name_valid


class LinearConv(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True):
        super(LinearConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = torch.nn.Linear(in_channels, out_channels, bias)

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

    def forward(self, x, edge_index, edge_weight=None):
        return torch.zeros([x.size(0), self.out_dim]).to(x.device)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


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


def gnn_map(gnn_name, in_dim, out_dim, concat=False, bias=True) -> Module:
    '''

    :param gnn_name:
    :param in_dim:
    :param out_dim:
    :param concat: for gat, concat multi-head output or not
    :return: GNN model
    '''
    if gnn_name == "gat_8":
        return GATConv(in_dim, out_dim, 8, concat=concat, bias=bias)
    elif gnn_name == "gat_6":
        return GATConv(in_dim, out_dim, 6, concat=concat, bias=bias)
    elif gnn_name == "gat_4":
        return GATConv(in_dim, out_dim, 4, concat=concat, bias=bias)
    elif gnn_name == "gat_2":
        return GATConv(in_dim, out_dim, 2, concat=concat, bias=bias)
    elif gnn_name in ["gat_1", "gat"]:
        return GATConv(in_dim, out_dim, 1, concat=concat, bias=bias)
    elif gnn_name == "gcn":
        return GCNConv(in_dim, out_dim)
    elif gnn_name == "cheb":
        return ChebConv(in_dim, out_dim, K=2, bias=bias)
    elif gnn_name == "sage":
        return SAGEConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "gated":
        return GatedGraphConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "arma":
        return ARMAConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "sg":
        return SGConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "linear":
        return LinearConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "zero":
        return ZeroConv(in_dim, out_dim, bias=bias)

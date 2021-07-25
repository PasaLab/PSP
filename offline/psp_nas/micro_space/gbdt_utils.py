import os
import copy
import numpy as np
import logging
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from collections import OrderedDict
import torchvision.transforms as transforms
from torch.nn import Module
from torch_geometric.nn.conv import *

from constant import *
from .micro_search_space import *

class GbdtUtils(object):
    def __init__(self, OPERATIONS, *args):
        self.OPERATIONS = OPERATIONS
        if len(args) > 0:
            self.num_nodes = args[0]

        self.REMAIN_FEA_LOWER_BOUND = [
            int(math.ceil(len(OPERATIONS[i]["value"]) / 2)) for i in range(len(self.OPERATIONS))
        ]
        # self.REMAIN_FEA_LOWER_BOUND.append(int(FEATURE_ENGINE_NUM / 2))
        # self.NONEED_PRUNED_FEATURE_ID = [0, 1, 14, 15, 16] if len(self.OPERATIONS[1]["value"]) == 12 else [0, 1, 10, 11, 12]
        self.NONEED_PRUNED_FEATURE_ID = []
        # 这里为了限制每个节点的激活函数不能去掉linear映射
        op_index = 0
        for i in range(self.num_nodes * 3):
            if i % 3 == 2:
                # cur_op_list = list(range(op_index, op_index + len(self.OPERATIONS[i]["value"])))
                linear_act_index = self.OPERATIONS[i]["value"].index('linear')
                self.NONEED_PRUNED_FEATURE_ID.append(op_index + linear_act_index)
            elif i % 3 == 0:
                # prev=1不能去掉，会影响ppnp算子
                # self.NONEED_PRUNED_FEATURE_ID.append(op_index + 1)
                        
                # 这里为了限制每个算子的前置节点不被去掉
                for j in range(len(self.OPERATIONS[i]["value"])):
                    self.NONEED_PRUNED_FEATURE_ID.append(op_index + j)
            op_index += len(self.OPERATIONS[i]["value"])


        # self.NONEED_PRUNED_FEATURE_ID = [0, 1, 14, 15, 16, 29, 30, 31, 32, 45, 46, 47, 48, 49] if len(self.OPERATIONS[1]["value"]) == 12 else [0, 1, 10, 11, 12, 21, 22, 23, 24]
        get_logger().info(f"NO NEED PRUNED FEATURE ID: {self.NONEED_PRUNED_FEATURE_ID}")
        self.PRUNE_NUM_FIRST = PRUNE_NUM_FIRST
        self.PRUNE_NUM_SECOND = PRUNE_NUM_SECOND
        self.PRUNE_NUM_LARGER = PRUNE_NUM_LARGER
        self.PRUNE_NUM = self.PRUNE_NUM_FIRST


        self.feature_engine_combine_num = int(pow(2, len(self.OPERATIONS[-1]["value"])))

        self.select_prob = copy.deepcopy(self.OPERATIONS)

    def demical_to_bin(self, x):
        _x = x
        res = []
        while _x > 0:
            res.append(_x % 2)
            _x = int(_x / 2)
        while len(res) < len(self.OPERATIONS[-1]["value"]):
            res.append(0)
        res.reverse()
        return res

    def generate_arch(self, n):
        def _get_arch():
            feature_dim = 0
            hidden_dim = 1
            while True:
                arch = []
                op_index = 0
                linear_act_index = ACT_LIST.index('linear')

                for op in self.OPERATIONS:
                    cur_op_num = len(op["value"]) if op["name"] != "feature_engine" else self.feature_engine_combine_num
                    # cur_op = np.random.randint(0, cur_op_num)
                    cur_op = random.randint(0, cur_op_num - 1)
                    
                    if op['name'][:3] == 'gnn' and ('identity' in op["value"]):
                        identity_index = op["value"].index('identity')

                    # 做关于第一个节点的限制: 如果选择了ppnp，那么前置节点不能是原始特征，一定是经过transform的
                    # if op_index == 1:
                    if op_index < self.num_nodes * 3 and op_index % 3 == 1 and op["value"][cur_op] == 'appnp': # or GNN_LIST[cur_op] == 'gat_relation':
                        prev_index = arch[-1]
                        if prev_index == 0:
                            prev_index = random.randint(1, len(self.OPERATIONS[op_index - 1]['value']) - 1)
                            arch[-1] = prev_index

                    # 如果当前op选择了identity，设置激活函数也选择linear，相当于这一层轮空
                    if (0 < op_index < self.num_nodes * 3) and (op_index % 3 == 2):
                        if arch[-1] == identity_index:
                            cur_op = linear_act_index

                    if op["name"] == "feature_engine":
                        while cur_op == 0:
                            # cur_op = np.random.randint(0, cur_op_num)
                            cur_op = random.randint(0, cur_op_num - 1)

                    # get_logger().info(f"cur_op_selected: {cur_op}")
                    arch.append(cur_op)
                    op_index += 1

                flag = 0
                # 这里是为了统计选择identity算子的nodes一共有几个，保证至少聚合层最少1个
                identity_sum = 0
                for i in range(1, self.num_nodes * 3, 3):
                    if self.OPERATIONS[i]['value'][arch[i]] == 'identity':
                        identity_sum += 1
                # get_logger().info(f"identity_sum: {identity_sum}")
                if identity_sum < self.num_nodes:
                    flag += 1
                
                # 有可能会出现ppnp算子的前置节点输出维度是feature_num

                out_dim = [0] * (self.num_nodes + 2)
                out_dim[0] = feature_dim
                out_dim[1] = hidden_dim
                # out_dim[2] = hidden_dim

                proper_flag = True
                for i in range(self.num_nodes * 3):
                    if i % 3 == 0:
                        prev = arch[i]
                        op = arch[i + 1]
                        if GNN_LIST[op] != 'identity' and GNN_LIST[op] != 'appnp':
                            out_dim[i // 3 + 2] = hidden_dim
                        elif GNN_LIST[op] == 'identity':
                            out_dim[i // 3 + 2] = out_dim[prev]
                        elif GNN_LIST[op] == 'appnp':
                            out_dim[i // 3 + 2] = out_dim[prev]
                            if out_dim[i // 3 + 2] == feature_dim:
                                proper_flag = False
                                break
                
                if proper_flag:
                    flag += 1
                
                if flag == 2:
                    break

            return arch

        archs = []
        while len(archs) < n:
            arch = _get_arch()
            # get_logger().info("")
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
                tmp = self.demical_to_bin(op)
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
            feature_dim = 0
            hidden_dim = 1
            while True:
                arch = []
                fea_base = 0
                linear_act_index = ACT_LIST.index('linear')
                for i in range(len(self.OPERATIONS)):
                    candidates = []
                    fea_base += len(self.OPERATIONS[i - 1]["value"]) if i > 0 else 0
                    # get_logger().info(fea_base)
                    if self.OPERATIONS[i]["name"] == "feature_engine":
                        for j in range(self.feature_engine_combine_num):
                            mutil_hot_feature = self.demical_to_bin(j)
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
                    gen_op = random.choice(candidates)

                    op = self.OPERATIONS[i]
                    cur_op_num = len(op["value"]) if op["name"] != "feature_engine" else self.feature_engine_combine_num

                    if op['name'][:3] == 'gnn' and ('identity' in op["value"]):
                        identity_index = op["value"].index('identity')

                    # 做关于第一个节点的限制: 如果选择了ppnp，那么前置节点不能是原始特征，一定是经过transform的
                    if i < self.num_nodes * 3 and i % 3 == 1 and op["value"][gen_op] == 'appnp': # or GNN_LIST[cur_op] == 'gat_relation':
                        prev_index = arch[-1]
                        while prev_index == 0:
                            prev_index = random.choice(last_candidates)
                        arch[-1] = prev_index

                    # 如果当前op选择了identity，设置激活函数也选择linear，相当于这一层轮空
                    if (0 < i < self.num_nodes * 3) and (i % 3 == 2):
                        if arch[-1] == identity_index:
                            gen_op = linear_act_index

                    if self.OPERATIONS[i]["name"] == "feature_engine":
                        while gen_op == 0:
                            # gen_op = np.random.choice(candidates)
                            gen_op = random.choice(candidates)

                    last_candidates = copy.deepcopy(candidates)
                    arch.append(gen_op)
                
                flag = 0
                # 这里是为了统计选择identity算子的nodes一共有几个，保证至少聚合层最少1个
                identity_sum = 0
                for i in range(1, self.num_nodes * 3, 3):
                    if self.OPERATIONS[i]['value'][arch[i]] == 'identity':
                        identity_sum += 1
                # get_logger().info(f"identity_sum: {identity_sum}")
                if identity_sum < self.num_nodes:
                    flag += 1
                
                # 有可能会出现ppnp算子的前置节点输出维度是feature_num

                out_dim = [0] * (self.num_nodes + 2)
                out_dim[0] = feature_dim
                out_dim[1] = hidden_dim
                # out_dim[2] = hidden_dim

                proper_flag = True
                for i in range(self.num_nodes * 3):
                    if i % 3 == 0:
                        prev = arch[i]
                        op = arch[i + 1]
                        if GNN_LIST[op] != 'identity' and GNN_LIST[op] != 'appnp':
                            out_dim[i // 3 + 2] = hidden_dim
                        elif GNN_LIST[op] == 'identity':
                            out_dim[i // 3 + 2] = out_dim[prev]
                        elif GNN_LIST[op] == 'appnp':
                            out_dim[i // 3 + 2] = out_dim[prev]
                            if out_dim[i // 3 + 2] == feature_dim:
                                proper_flag = False
                                break
                
                if proper_flag:
                    flag += 1
                
                if flag == 2:
                    break

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
        mutil_hot_feature = self.demical_to_bin(arch[-1])
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


def get_logger(name=__file__, level=logging.INFO):
    logger = logging.getLogger(name)

    if getattr(logger, '_init_done__', None):
        logger.setLevel(level)
        return logger

    logger._init_done__ = True
    logger.propagate = False
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(0)

    del logger.handlers[:]
    logger.addHandler(handler)

    return logger


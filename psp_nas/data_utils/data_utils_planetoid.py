#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2020-05-07
import torch
import pandas as pd
import numpy as np
import torch_geometric.utils as gtils
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from utils import get_logger

from feat_engine import *

logger = get_logger()

FEAT_ENGINE = [
    svd_feature, edge_weights_feature, normalize_feature,
    degree_bins_feature# , prepredict_feature, lpa_feature
]

# FEAT_ENGINE = [
#     svd_feature,
#     edge_weights_feature, normalize_feature, degree_bins_feature
# ]


def format_autograph_data_planetoid(data, data_meta):
    data_dict = data
    # get x feature table
    x = data['fea_table']
    if x.shape[1] == 1:
        x = x.to_numpy()
        x = x.reshape(x.shape[0])
        data['fea_table'] = pd.concat([pd.get_dummies(x), data['fea_table']['node_index']], axis=1)
        x = np.array(data['fea_table'])
    else:
        x = x.drop('node_index', axis=1).to_numpy()

    # logger.info("x shape: {}".format(x.shape))
    x = torch.tensor(x, dtype=torch.float)

    # get edge_index, edge_weight
    df = data['edge_file']
    edges = df[['src_idx', 'dst_idx', 'edge_weight']].to_numpy()
    edge_index = edges[:, :2].astype(np.int)
    # transpose from [edge_num, 2] to [2, edge_num] which is required by PyG
    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
    edge_weight = edges[:, 2]
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32)

    undirected = gtils.is_undirected(edge_index)

    edge_index, edge_weight = gtils.sort_edge_index(edge_index, edge_weight)
    # logger.info(f"is undirected ? {undirected}")
    # logger.info(f"edge index {edge_index.shape}, edge weight {edge_weight.shape}")

    # get train/test mask
    num_nodes = x.size(0)
    y = torch.zeros(num_nodes, dtype=torch.long)
    train_inds = data['train_label'][['node_index']].to_numpy()
    train_y = data['train_label'][['label']].to_numpy()
    y[train_inds] = torch.tensor(train_y, dtype=torch.long)
    test_inds = data['test_label'][['node_index']].to_numpy()
    test_y = data['test_label'][['label']].to_numpy()
    y[test_inds] = torch.tensor(test_y, dtype=torch.long)

    train_indices = data['train_indices']
    val_indices = data['val_indices']
    test_indices = data['test_indices']

    logger.info(f"train num: {len(train_indices)}; valid num: {len(val_indices)}; test num:{len(test_indices)}")
    data = Data(x=x, edge_index=edge_index, y=y, edge_weight=edge_weight)

    data.num_nodes = num_nodes

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_indices] = 1
    data.train_indices = np.asarray(train_indices)
    data.train_mask = train_mask

    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[val_indices] = 1
    data.val_indices = val_indices
    data.val_mask = val_mask

    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[test_indices] = 1
    data.test_mask = test_mask
    data.test_indices = np.asarray(test_indices)

    meta_info = {
        'num_nodes': num_nodes,
        'num_class': data_meta['n_class'],
        'is_undirected': undirected
    }
    data_dict['feat_df'] = data_dict['fea_table']

    logger.info(f"feat_df:\n {data_dict['feat_df']}")

    data_dict['edge_df'] = data_dict['edge_file'][['src_idx', 'dst_idx', 'edge_weight']]
    data_dict['meta_info'] = meta_info
    drop_n_unique(**data_dict)

    features = [data_dict['feat_df']]
    global FEAT_ENGINE
    for feature in FEAT_ENGINE:
        features.append(feature(**data_dict))

    if data_dict['feat_df'].shape[1] == 1:
        x_numpy = data_dict['feat_df'].to_numpy()
        x_numpy = x_numpy.reshape(x_numpy.shape[0])
        data_dict['feat_df'] = pd.concat([pd.get_dummies(x_numpy), data_dict['feat_df']['node_index']], axis=1)
    else:
        data_dict['feat_df'] = data_dict['feat_df'].drop('node_index', axis=1)

    features[0] = data_dict['feat_df']

    # logger.info(f"feat_df:\n {data_dict['feat_df']}")
    #
    logger.info(f"features[0]:\n {features[0]}")

    for i in range(len(features)):
        logger.info(f"feature id: {i}; feature shape: {features[i].shape}")

    return data, features

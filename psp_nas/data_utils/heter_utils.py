import os
import re
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.model_selection import ShuffleSplit
from .utils import sys_normalized_adjacency,sparse_mx_to_torch_sparse_tensor
import pickle as pkl
import sys
import networkx as nx
import numpy as np
import scipy.sparse as sp
import pandas as pd
from collections import Counter

import torch_geometric.utils as gtils
from torch_geometric.data import Data

# from gbdt_utils import get_logger
from utils import get_logger

from feat_engine import *

logger = get_logger()

FEAT_ENGINE = [
    svd_feature,
    edge_weights_feature, normalize_feature, degree_bins_feature
    # ,prepredict_feature, lpa_feature
]

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    rowsum = (rowsum==0)*1+rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def form_data_heter(adj, features, labels, train_mask, val_mask, test_mask):
    # 转换成pyg的edge_index格式
    edge_index = adj.coalesce().indices()
    edge_file = pd.DataFrame(edge_index.cpu().numpy().T, columns=['src_idx', 'dst_idx'])
    edge_file['edge_weight'] = 1.0
    fea_table = pd.DataFrame(features.cpu().numpy()).reset_index().rename(columns={'index': 'node_index'})

    # get x feature table
    x = fea_table
    if x.shape[1] == 1:
        x = x.to_numpy()
        x = x.reshape(x.shape[0])
        fea_table = pd.concat([pd.get_dummies(x), fea_table['node_index']], axis=1)
        x = np.array(fea_table)
    else:
        x = x.drop('node_index', axis=1).to_numpy()

    x = torch.tensor(x, dtype=torch.float)

    edges = edge_file[['src_idx', 'dst_idx', 'edge_weight']].to_numpy()
    edge_weight = edges[:, 2]
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32)

    undirected = gtils.is_undirected(edge_index)
    edge_index, edge_weight = gtils.sort_edge_index(edge_index, edge_weight)

    y = torch.tensor(labels, dtype=torch.long)

    # form pyg format dataloader
    data = Data(x=x, edge_index=edge_index, y=y, edge_weight=edge_weight)

    data.num_nodes = x.size(0)
    
    data_dict = {}
    
    train_indices = [i for i in range(len(train_mask)) if train_mask[i] == True]
    val_indices = [i for i in range(len(val_mask)) if val_mask[i] == True]
    test_indices = [i for i in range(len(test_mask)) if test_mask[i] == True]

    data_dict['train_indices'] = train_indices
    data_dict['val_indices'] = val_indices
    data_dict['test_indices'] = test_indices

    # 处理划分
    data.train_mask = torch.BoolTensor(train_mask)
    data.val_mask = torch.BoolTensor(val_mask)
    data.test_mask = torch.BoolTensor(test_mask)
    
    meta_info = {
        'num_nodes': data.num_nodes,
        'num_class': len(Counter(data.y.cpu().numpy())),
        'is_undirected': undirected
    }

    train_label = pd.DataFrame(
                    {
                        'node_index': np.where(data.train_mask | data.val_mask)[0],
                        'label': data.y[data.train_mask | data.val_mask].cpu().numpy()})
    logger.info(f"train_label: {train_label}")

    test_label = pd.DataFrame(
        {'node_index': np.where(data.test_mask)[0], 'label': data.y[data.test_mask].cpu().numpy()})

    data_dict['train_label'] = train_label
    data_dict['test_label'] = test_label

    data_dict['feat_df'] = fea_table

    data_dict['edge_df'] = edge_file[['src_idx', 'dst_idx', 'edge_weight']]
    data_dict['meta_info'] = meta_info

    drop_n_unique(**data_dict)

    features = [data_dict['feat_df']]
    global FEAT_ENGINE
    for feature in FEAT_ENGINE:
        # logger.info(f"feature_engin: {feature}")
        features.append(feature(**data_dict))
        # logger.info(f"features: {features}\n")
        
    if data_dict['feat_df'].shape[1] == 1:
        x_numpy = data_dict['feat_df'].to_numpy()
        x_numpy = x_numpy.reshape(x_numpy.shape[0])
        data_dict['feat_df'] = pd.concat([pd.get_dummies(x_numpy), data_dict['feat_df']['node_index']], axis=1)
    else:
        data_dict['feat_df'] = data_dict['feat_df'].drop('node_index', axis=1)

    features[0] = data_dict['feat_df']

    # logger.info(f"feat_df:\n {data_dict['feat_df']}")
    
    for i in range(len(features)):
        logger.info(f"feature id: {i}; feature shape: {features[i].shape}")

    return data, features


def process_heter(dataset, dataset_dir_path, spilt_str):
    graph_adjacency_list_file_path = os.path.join(dataset_dir_path, 'new_data', dataset, 'out1_graph_edges.txt')
    graph_node_features_and_labels_file_path = os.path.join(dataset_dir_path, 'new_data', dataset,
                                                                'out1_node_feature_label.txt')
    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}
    
    with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
        graph_node_features_and_labels_file.readline()
        for line in graph_node_features_and_labels_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 3)
            assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
            graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
            graph_labels_dict[int(line[0])] = int(line[2])

    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                            label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                            label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))

    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    features = np.array(
        [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array(
        [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
    features = preprocess_features(features)

    # adj = sys_normalized_adjacency(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    with np.load(spilt_str) as splits_file:
        train_mask = splits_file['train_mask']
        val_mask = splits_file['val_mask']
        test_mask = splits_file['test_mask']
    
    num_features = features.shape[1]
    num_labels = len(np.unique(labels))
    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    data, features = form_data_heter(adj, features, labels, train_mask, val_mask, test_mask)

    return data, features
    # g = adj

    # with np.load(splits_file_path) as splits_file:
    #     train_mask = splits_file['train_mask']
    #     val_mask = splits_file['val_mask']
    #     test_mask = splits_file['test_mask']
    
    # num_features = features.shape[1]
    # num_labels = len(np.unique(labels))
    # assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

    # features = th.FloatTensor(features)
    # labels = th.LongTensor(labels)
    # train_mask = th.BoolTensor(train_mask)
    # val_mask = th.BoolTensor(val_mask)
    # test_mask = th.BoolTensor(test_mask)

    # g = sys_normalized_adjacency(g)
    # g = sparse_mx_to_torch_sparse_tensor(g)

    # return g, features, labels, train_mask, val_mask, test_mask, num_features, num_labels
import os
import sys
import glob
import time
import copy
import random
import numpy as np

import lightgbm
import shap

from micro_space import gbdt_utils
from constant import *
from micro_space.gbdt_utils import get_logger

logger = get_logger()

def softmax(x):
    """Compute the softmax in a numerically stable way."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def prune_uni_search_space(gbdtutils, bst, seqs, accs, pruned_operations, each_feature_remained, *args):
    xs = np.array(list(map(gbdtutils.convert_to_features, seqs)))
    ys = np.array(accs)

    old_prune_count = len(pruned_operations)
    logger.info(f"old prune count: {old_prune_count}")

    feature_names = bst.feature_name()
    bst.params['objective'] = 'regression'
    explainer = shap.TreeExplainer(bst)

    feature_shap_values = []
    shap_values = explainer.shap_values(xs)

    pos_shap_value_dict = {}

    for feature_id in range(len(feature_names)):
        feature_name = feature_names[feature_id]
        pos = []
        for i in range(len(xs)):
            cond = gbdtutils.parse_feature_to_cond(feature_name, xs[i])
            if cond:
                pos.append(i)
        pos_shap_value = np.mean(shap_values[pos, feature_id], axis=0) if len(pos) > 0 else 0
        feature_shap_values.append((feature_id, pos_shap_value))
        pos_shap_value_dict[f"{feature_id}"] = shap_values[pos, feature_id]
    
    ## 打印一下信息
    current_feature_id = 0
    while True:
        fea_i = gbdtutils.valid_fea_num_trans_to_i(current_feature_id)
        for i in range(len(gbdtutils.OPERATIONS[fea_i]['value'])):
            logger.info(f"{gbdtutils.OPERATIONS[fea_i]['name']}: {gbdtutils.OPERATIONS[fea_i]['value'][i]} {feature_shap_values[current_feature_id][1]}")
            current_feature_id += 1
        if current_feature_id >= len(feature_shap_values):
            break
        
    logger.info(f"sort the feature importance:")
    feature_shap_values_descent = sorted(feature_shap_values, key=lambda i:i[1], reverse=True)
    for feature_id, pos_shap_value in feature_shap_values_descent:
        fea_i = gbdtutils.valid_fea_num_trans_to_i(feature_id)
        value_i = gbdtutils.valid_fea_num_trans_to_str(feature_id)
        logger.info(f"{gbdtutils.OPERATIONS[fea_i]['name']}: {value_i} {pos_shap_value}")

    ## 统计一下每个特征类的信息
    operations = gbdtutils.OPERATIONS
    each_feature_class_importance = []
    fea_index = 0
    for i in range(len(operations) - 1):
        value_temp = []
        for j in range(len(operations[i]['value'])):
            value_temp.append((j, feature_shap_values[fea_index + j][1]))
        value_temp = sorted(value_temp, key=lambda i:i[1], reverse=True)
        each_feature_class_importance.append(value_temp)
        fea_index += len(operations[i]['value'])

    # 加特征工程的那一个选项
    # TODO 多项特征工程
    each_feature_class_importance.append([(1, feature_shap_values[-1][1])])
    
    feature_shap_values = sorted(feature_shap_values, key=lambda i:i[1])

    for feature_id, pos_shap_value in feature_shap_values:
        if feature_id in pruned_operations:
            continue
        if feature_id in gbdtutils.NONEED_PRUNED_FEATURE_ID:
            continue

        fea_i = gbdtutils.valid_fea_num_trans_to_i(feature_id)
        logger.info(f"feature_id: {feature_id}  <=>   fea_i: {fea_i}")
        if pos_shap_value < 0:
            if each_feature_remained[fea_i] - 1 < gbdtutils.REMAIN_FEA_LOWER_BOUND[fea_i]:
                logger.info(f"if del {feature_id}, remained features is too few")
                continue

            if len(pruned_operations) + 1 - old_prune_count <= gbdtutils.PRUNE_NUM:
                logger.info(f"{feature_id} is pruned successfully")
                pruned_operations[feature_id] = True
                each_feature_remained[fea_i] -= 1

            logger.info(f"current pruned operations: {pruned_operations}")

        prune_count = len(pruned_operations) - old_prune_count

        logger.info(f"cur prune_count: {prune_count}\n")
        if prune_count >= gbdtutils.PRUNE_NUM:
            logger.info("prune num is larger than PRUNE_NUM")
            break
        
    return pos_shap_value_dict, shap_values, each_feature_class_importance

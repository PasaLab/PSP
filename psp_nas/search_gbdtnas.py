import os
import sys
import glob
import time
import copy
import random
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import lightgbm as lgb
import shap
import time
import itertools

from constant import *
from gbdt_prune import prune_uni_search_space
from micro_space.micro_model_manager import MicroModelManager

from micro_space.gbdt_utils import get_logger, GbdtUtils
from macro_space.macro_utils import MacroUtils

logger = get_logger()


def build_args():
    parser = argparse.ArgumentParser(description='gbdt_nas')
    register_default_args(parser)
    args = parser.parse_args()

    return args


def register_default_args(parser):
    parser.add_argument("--dataset", type=str, default="a", required=False,
                            help="The input dataset.")
    parser.add_argument("--logger_path", type=str, default="a", required=False)
    parser.add_argument("--use_early_stop", type=str, default="0", required=False)
    parser.add_argument('--search_mode', type=str, default='micro')
    parser.add_argument('--submanager_log_file', type=str, default=f"sub_manager_logger_file_{time.time()}.txt")
    parser.add_argument('--shap_values_file', type=str, default=f"shap_values_file_{time.time()}.npy")
    parser.add_argument('--inputs_file', type=str, default=f"inputs_file_{time.time()}.npy")
    parser.add_argument('--SEED', type=int, default=123)
    parser.add_argument('--split_type', type=str, default="standard")
    # parser.add_argument('--search_space', type=str, default='micro')
    parser.add_argument('--prune_method', type=str, default='uni')
    parser.add_argument('--num_cells', type=int, default=2)
    parser.add_argument('--num_nodes', type=int, default=2)
    parser.add_argument('--split_id', type=int, default=0)

def init_proc(args):
    if not torch.cuda.is_available():
        logger.info('no gpu device available')
        sys.exit(1)

    SEED = args.SEED

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True

def get_feature_name(gbdtutils):
    feature_names = []
    for i in range(len(gbdtutils.OPERATIONS) - 1):
        op = gbdtutils.OPERATIONS[i]
        for j in range(len(op["value"])):
            feature_names.append("op {} use {}".format(i, j))
    for j in range(len(gbdtutils.OPERATIONS[-1]["value"])):
        feature_names.append("op {} use {}".format(len(gbdtutils.OPERATIONS) - 1, j))
    return feature_names


def train_and_valid(gbdtutils, arch_pool, model_manager, targeted_gen_num):
    # arch: [0, 1, 2, 0, 1 ……]      each selected op: (2: index 2 in "prev_1" list)
    valid_pool_acc = []
    test_pool_acc = []
    filtered_arch_pool = []
    arch_id = 0
    for arch in arch_pool:
        gnn_structure = gbdtutils.transform_to_valid_value(arch)
        actual_action = {
            "action": gnn_structure[: -6],
            "hyper_param": gnn_structure[-6: -1],
            "feature_engine": gnn_structure[-1]
        }
        try:
            logger.info(f"arch: {actual_action}")
            val_score_list = []
            test_score_list = []
            for train_round in range(3):
                val_score, test_score, stop_epoch = model_manager.train(actual_action)
                if val_score == 0:
                    break
                val_score_list.append(val_score)
                test_score_list.append(test_score)
            if val_score == 0:
                continue

            filtered_arch_pool.append(arch)

            logger.info(
                f"arc_id: {arch_id}, {actual_action}, val_score:{np.mean(val_score_list)} +/- {np.std(val_score_list)}, test_score:{np.mean(test_score_list)} +/- {np.std(test_score_list)}, stop_epoch:{stop_epoch}, val_list: {val_score_list}, test_list: {test_score_list}")

            valid_pool_acc.append(np.mean(val_score_list))
            test_pool_acc.append(np.mean(test_score_list))

            arch_id += 1
            if len(filtered_arch_pool) >= targeted_gen_num:
                break
        except RuntimeError as e:
            if "cuda" in str(e) or 'CUDA' in str(e):                        # avoid CUDA out of memory
                logger.info(f"we met cuda OOM; error message: {e}")
            else:
                raise e

    logger.info("#### training child model over ####")

    return filtered_arch_pool, valid_pool_acc, test_pool_acc


def evaluate(actual_action, model_manager, train_epoch=EPOCHS, retrain_stage=None):
    try:
    
        val_score, test_score, stop_epoch = model_manager.train(actual_action, retrain_stage, train_epoch=train_epoch)
        logger.info(f"{actual_action}, val_score:{val_score}, test_score:{test_score}, stop_epoch:{stop_epoch}")
    except RuntimeError as e:
        if "cuda" in str(e) or 'CUDA' in str(e):  # avoid CUDA out of memory
            logger.info(f"we met cuda OOM; error message: {e}")
            val_score = 0
            test_score = 0
        else:
            raise e
    return val_score, test_score


def train_controller(gbdtutils, params, feature_name, train_input, train_target, num_boost_round):
    logger.info('Train data: {}'.format(len(train_input)))

    ### 网格搜索调参
    train_x = np.array(list(map(gbdtutils.convert_to_features, train_input)))
    train_y = np.array(train_target)

    model_lgb = lgb.LGBMRegressor(objective='regression',
                                metric='l2', colsample_bytree = 0.9, subsample = 0.8, subsample_freq=2, verbose=-1)

    from sklearn.model_selection import GridSearchCV
    params_test = {
        'learning_rate': [0.01, 0.05, 0.001, 0.1, 0.005],
        'num_leaves': [15, 31, 63, 127, 170],
        'min_child_samples': list(range(7, 26)),
        'n_estimators': [50, 80, 100, 150, 200, 300]
    }

    gsearch = GridSearchCV(estimator=model_lgb, param_grid=params_test, scoring='neg_mean_squared_error', cv=5, verbose=-1, n_jobs=8)
    gsearch.fit(train_x, train_y)
    grid_parmas = gsearch.cv_results_['params']
    grid_scores = gsearch.cv_results_['mean_test_score']
    grid_rank = gsearch.cv_results_['rank_test_score']
#     for params, scores, rank in zip(grid_parmas, grid_scores, grid_rank):
#         print(f"{params} {scores} {rank}")
    logger.info(f"best_parmas: {gsearch.best_params_}")
    params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l2'},
            'min_data_in_leaf': gsearch.best_params_['min_child_samples'],
            'num_leaves': gsearch.best_params_['num_leaves'],
            'learning_rate': gsearch.best_params_['learning_rate'],
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 2,
            'verbose': -1
        }

    lgb_train = lgb.Dataset(train_x, train_y)
    gbm = lgb.train(params, lgb_train, feature_name=feature_name, num_boost_round=gsearch.best_params_['n_estimators'])

    return gbm

def calc_predictor_ranking_rate(new_arch_valid):
    inverse_num = 0
    arch_num = len(new_arch_valid)
    for i in range(arch_num):
        for j in range(i + 1, arch_num):
            if new_arch_valid[i] < new_arch_valid[j]:
                inverse_num += 1
    total_comp_pair = arch_num * (arch_num - 1) // 2
    return inverse_num / total_comp_pair

def main():

    args = build_args()

    logger.info(f"args: {args}")

    init_proc(args)

    logger.info(f"args use early stop: {args.use_early_stop}")

    logger.info(f"N:{CONTROLLER_N}; M:{CONTROLLER_M}; K:{CONTROLLER_K}; SEED:{args.SEED}")

    # construct Model Manager
    model_manager = MicroModelManager(args)
    OPERATIONS = model_manager.operations

    gbdtutils = GbdtUtils(OPERATIONS, args.num_nodes)

    feature_name = get_feature_name(gbdtutils)

    prune_func = prune_uni_search_space

    child_arch_pool = None

    if child_arch_pool is None:
        logger.info('Architecture pool is not provided, randomly generating now')
        child_arch_pool = gbdtutils.generate_arch(CONTROLLER_N_LARGER)

    arch_pool = []
    arch_pool_valid_acc = []
    arch_pool_test_acc = []

    pruned_operations = {}

    each_feature_remained = [len(OPERATIONS[i]["value"]) for i in range(len(OPERATIONS))]

    logger.info(f"REMAIN_FEA_LOWER_BOUND: {gbdtutils.REMAIN_FEA_LOWER_BOUND}")

    final_arch_pool = []
    final_arch_pool_test_acc = []
    final_arch_pool_iteration = []

    final_arch_pool_valid_acc = []

    args_K = CONTROLLER_K

    relative_ranking_error = 0
    last_relative_ranking_error = 0

    ######################################## psp_nas train ############################################
    for controller_iteration in range(CONTROLLER_ITERATIONS + 1):

        logger.info(f"====================== search iter {controller_iteration} start ======================")

        targeted_gen_num = CONTROLLER_N if controller_iteration == 0 else args_K

        child_arch_pool, child_arch_pool_valid_acc, child_arch_pool_test_acc = train_and_valid(gbdtutils, child_arch_pool, model_manager, targeted_gen_num)

        arch_pool += child_arch_pool

        ### 衡量predictor的性能
        if controller_iteration >= 1:
            relative_ranking_error = calc_predictor_ranking_rate(child_arch_pool_valid_acc)       
            logger.info(f"relative_ranking_error: {relative_ranking_error * 100}%")

        final_arch_pool.extend(child_arch_pool)
        final_arch_pool_valid_acc.extend(child_arch_pool_valid_acc)
        final_arch_pool_iteration.extend([controller_iteration] * len(child_arch_pool))

        arch_pool_valid_acc += child_arch_pool_valid_acc
        arch_pool_test_acc += child_arch_pool_test_acc

        arch_pool_valid_acc_sorted_indices = np.argsort(arch_pool_valid_acc)[::-1]
        arch_pool = list(map(lambda x: arch_pool[x], arch_pool_valid_acc_sorted_indices))
        arch_pool_valid_acc = list(map(lambda x: arch_pool_valid_acc[x], arch_pool_valid_acc_sorted_indices))

        if controller_iteration == 0:
            gbdtutils.PRUNE_NUM = gbdtutils.PRUNE_NUM_FIRST
        elif controller_iteration == 1:
            gbdtutils.PRUNE_NUM = gbdtutils.PRUNE_NUM_SECOND
        else:
            gbdtutils.PRUNE_NUM = gbdtutils.PRUNE_NUM_LARGER

        inputs = arch_pool
        min_val = min(arch_pool_valid_acc)
        max_val = max(arch_pool_valid_acc)
        targets = list(map(lambda x: (x - min_val) / (max_val - min_val), arch_pool_valid_acc))

        logger.info('Train GBDT')

        logger.info("------ Train GBDT -------")
        logger.info(f"inputs len: {len(inputs)}")
        logger.info(f"targets len: {len(targets)}")

        logger.info(f"inputs: {inputs}")

        gbm = train_controller(gbdtutils, GBDT_PARAMS, feature_name, inputs, targets, CONTROLLER_NUM_BOOST_ROUND)

        shap_value_dict, shap_values, each_feature_class_importance = prune_func(gbdtutils, gbm, inputs, targets, pruned_operations, each_feature_remained, controller_iteration == CONTROLLER_ITERATIONS - 1)

        inputs_copy = copy.deepcopy(inputs)

        logger.info(f"pruned operations: {pruned_operations}")
        strr = ""
        for k, v in pruned_operations.items():
            strr += gbdtutils.valid_fea_num_trans_to_str(k)
            strr += " "
        logger.info(f"pruned operations: {strr}")

        strr = ""
        tmp = 0
        for i in range(len(OPERATIONS)):
            strr = strr + OPERATIONS[i]["name"] + ": "
            for j in range(len(OPERATIONS[i]["value"])):
                if tmp in pruned_operations and pruned_operations[tmp] is True:
                    tmp += 1
                    continue
                tmp += 1
                strr = strr + str(OPERATIONS[i]["value"][j]) + " "
            strr += "\n"

        logger.info(f"OPERATIONS after pruning:\n{strr}")

        if controller_iteration == CONTROLLER_ITERATIONS - 1:
            if relative_ranking_error > last_relative_ranking_error:
                break

        if controller_iteration == CONTROLLER_ITERATIONS:
            break

        # Ranking sampled candidates
        # 依据pruned_operations每层构造一个特征，随机sample的M个arch，经过gdbt的打分后选择前K个archs，放进new_arch这个池子里
        if controller_iteration < CONTROLLER_ITERATIONS - 1:
            M_gen = CONTROLLER_M 
        else:
            M_gen = CONTROLLER_M_LARGER
            
        random_arch = gbdtutils.generate_constrained_arch(M_gen, pruned_operations, args.prune_method)
        logger.info('Totally {} archs sampled from the search space'.format(len(random_arch)))
        random_arch_features = np.array(list(map(gbdtutils.convert_to_features, random_arch)))

        start_time = time.time()
        random_arch_pred = gbm.predict(random_arch_features, num_iteration=gbm.best_iteration)
        logger.info(f"gbm predict time: {time.time() - start_time}")

        sorted_indices = np.argsort(random_arch_pred)[::-1]
        random_arch = [random_arch[i] for i in sorted_indices]

        if controller_iteration >= 1:
            args_K += PER_ROUND_ADD

        new_arch = []
        for arch in random_arch:
            if arch in arch_pool:
                continue
            new_arch.append(arch)
            if len(new_arch) >= args_K * 3:
                break

        logger.info("Generate %d new archs", len(new_arch))
        child_arch_pool = new_arch  # + arch_pool[:200]

        last_relative_ranking_error = copy.deepcopy(relative_ranking_error)

    logger.info('Finish Searching')

    #################################### print search info ##################################
    logger.info("====================== search result =======================")

    shap_values = np.array(shap_values)
    shap_file_path = os.path.join(args.logger_path, args.search_mode + args.shap_values_file)
    inputs_file_path = os.path.join(args.logger_path, args.search_mode + args.inputs_file)

    np.save(shap_file_path, shap_values)
    np.save(inputs_file_path, inputs_copy)

    logger.info(f"final_arch_pool len: {len(final_arch_pool)}")
    logger.info(f"final_arch_pool_valid_acc len: {len(final_arch_pool_valid_acc)}")
    logger.info(f"final_arch_pool_iteration len: {len(final_arch_pool_iteration)}")

    logger.info(f"final_arc_pool_valid_acc Not sorted: {final_arch_pool_valid_acc}")

    final_arch_pool_valid_acc_sorted_indices = np.argsort(final_arch_pool_valid_acc)[::-1]
    final_arch_pool = list(map(lambda x: final_arch_pool[x], final_arch_pool_valid_acc_sorted_indices))
    final_arch_pool_valid_acc = list(map(lambda x: final_arch_pool_valid_acc[x], final_arch_pool_valid_acc_sorted_indices))
    final_arch_pool_iteration = list(map(lambda x: final_arch_pool_iteration[x], final_arch_pool_valid_acc_sorted_indices))

    logger.info(f"final_arc_pool_valid_acc sorted: {final_arch_pool_valid_acc}")


    res_str = f"\n=============================== top {TOP_K} arch being valit ====================================\n"
    iteration_num = [0] * (CONTROLLER_ITERATIONS + 2)
    iteration_acc = [0] * (CONTROLLER_ITERATIONS + 2)

    best_structure = ""
    best_val_score = 0
    best_test_score = 0

    model_manager.is_use_early_stop = False

    for i in range(TOP_K):
        arch = final_arch_pool[i]
        logger.info(f"not re valid --- arc: {arch}; valid_acc: {final_arch_pool_valid_acc[i]}")

        gnn_structure = gbdtutils.transform_to_valid_value(arch)

        actual_action = {
            "action": gnn_structure[: -6],
            "hyper_param": gnn_structure[-6: -1],
            "feature_engine": gnn_structure[-1]
        }
        actual_action_copy = copy.deepcopy(actual_action)

        val_scores_list = []
        test_scores_list = []
        for j in range(EACH_TOPK_RETRAIN_ROUND):
            logger.info(f"######### arch {i}, validate iter {j} ##########")
            val_acc, test_acc = evaluate(actual_action_copy, model_manager, retrain_stage="revalid")
            actual_action_copy = copy.deepcopy(actual_action)
            if val_acc == 0:
                continue
            val_scores_list.append(val_acc)
            test_scores_list.append(test_acc)

        tmp_val_score = np.mean(val_scores_list)
        tmp_test_score = np.mean(test_scores_list)

        logger.info(f"-------------------- arch {i}: {actual_action}, avg val score: {tmp_val_score}, avg test score: {tmp_test_score}--------------------------")
        if tmp_val_score > best_val_score:
            best_val_score = tmp_val_score
            best_structure = actual_action

        res_str = res_str + "arch: " + str(actual_action) + "\t" + "val acc: " + str(
            tmp_val_score) + "\t" + "test acc: " + str(tmp_test_score) + "\t" + "search iter: " + str(final_arch_pool_iteration[i])
        res_str += "\n"
        iteration_num[final_arch_pool_iteration[i]] += 1
        iteration_acc[final_arch_pool_iteration[i]] += best_val_score

    logger.info(res_str)

    strr = "\n=================================== iter and test score statistics ==================================\n"
    for i in range(CONTROLLER_ITERATIONS + 2):
        if iteration_num[i] == 0:
            continue
        avg_val_acc = iteration_acc[i] * 1.0 / iteration_num[i]
        iter_ratio = iteration_num[i] * 1.0 / TOP_K
        strr = strr + "iter: " + str(i) + " #### " + "iter ratio: " + str(iter_ratio) + " #### " + "avg val acc: " + str(avg_val_acc) + "\n"

    logger.info(strr)

    logger.info("================================= last result ====================================")

    logger.info("best structure:" + str(best_structure))
    best_structure_copy = copy.deepcopy(best_structure)
    # train from scratch to get the final score
    test_scores_list = []
    test_acc_percent = []
    valid_acc_percent = []
    for i in range(BEST_ARCH_RETRAIN_ROUND):
        val_acc, test_acc = evaluate(best_structure_copy, model_manager, train_epoch=BEST_ARC_RETRAIN_EPOCH, retrain_stage="bst_retrain")
        best_structure_copy = copy.deepcopy(best_structure)
        if test_acc == 0:
            continue
        test_scores_list.append(test_acc)
        test_acc_percent.append(test_acc * 100.0)
        valid_acc_percent.append(val_acc * 100.0)
    test_acc_percent.sort()
    valid_acc_percent.sort()
    logger.info(f"\nbest results: {best_structure}"
                f"\nbest acc(%): {np.mean(test_acc_percent[5:-5]):.8f} var(%): {np.var(test_acc_percent[5:-5])} std(%): {np.std(test_acc_percent[5:-5])}"
                f"\nbest valid acc(%): {np.mean(valid_acc_percent[5:-5]):.8f} var(%): {np.var(valid_acc_percent[5:-5])} std(%): {np.std(valid_acc_percent[5:-5])}")

if __name__ == '__main__':
    main()

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
import copy

SEED = 123

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True

sys.path.append("../psp_nas/")

from constant import *
from micro_space.micro_model_manager import MicroModelManager
from utils import get_logger

logger = get_logger()

def build_args():
    parser = argparse.ArgumentParser(description='gbdt_nas_evaluate')
    register_default_args(parser)
    args = parser.parse_args()

    return args


def register_default_args(parser):
    parser.add_argument("--dataset", type=str, default="a", required=False,
                            help="The input dataset.")
    parser.add_argument('--split_id', type=int, default=0)
    parser.add_argument("--logger_path", type=str, default="a", required=False)
    parser.add_argument("--use_early_stop", type=str, default="1", required=False)
    parser.add_argument('--search_mode', type=str, default='micro')
    parser.add_argument('--submanager_log_file', type=str, default=f"sub_manager_logger_file_{time.time()}.txt")
    parser.add_argument('--shap_values_file', type=str, default=f"shap_values_file_{time.time()}.npy")
    parser.add_argument('--inputs_file', type=str, default=f"inputs_file_{time.time()}.npy")
    parser.add_argument('--SEED', type=int, default=123)
    parser.add_argument('--split_type', type=str, default="full_supervised")
    # parser.add_argument('--search_space', type=str, default='micro')
    parser.add_argument('--num_cells', type=int, default=2)
    parser.add_argument('--num_nodes', type=int, default=2)

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
    # cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True

    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED)

    os.environ['PYTHONHASHSEED'] = str(SEED)

def evaluate(actual_action, model_manager, train_epoch=EPOCHS, retrain_stage=None):
    try:
        val_score, test_score, stop_epoch = model_manager.train(actual_action, retrain_stage, train_epoch=train_epoch)
        logger.info(f"{actual_action}, val_score:{val_score}, test_score:{test_score}, stop_epoch:{stop_epoch};")
    except RuntimeError as e:
        if "cuda" in str(e) or 'CUDA' in str(e):  # avoid CUDA out of memory
            logger.info(f"we met cuda OOM; error message: {e}")
            val_score = 0
            test_score = 0
        else:
            raise e
    return val_score, test_score


def main():

    args = build_args()

    # construct Model Manager
    model_manager = MicroModelManager(args)

    
    structure_list = {
        "standard": {
            'Cora': {'action': [1, 'cos', 'tanh', 2, 'appnp', 'elu', 'add'], 'hyper_param': [5e-4, 0.8, 1e-3, 5e-3, 512], 'feature_engine': ['origin']},
            'Citeseer': {'action': [1, 'appnp', 'linear', 1, 'appnp', 'relu', 'max'], 'hyper_param': [1e-2, 0.3, 1e-3, 5e-3, 64], 'feature_engine': ['origin']},
            'Pubmed': {'action': [1, 'appnp', 'elu', 2, 'appnp', 'tanh', 'add'], 'hyper_param': [1e-2, 0.0, 5e-3, 1e-3, 256], 'feature_engine': ['origin']},
        },

        "full_supervised":{
            'Cora': {'action': [1, 'appnp', 'tanh', 1, 'appnp', 'relu', 'add'], 'hyper_param': [0.005, 0.8, 0.001, 0.0005, 512], 'feature_engine': ['origin']},
            'Citeseer': {'action': [1, 'appnp', 'tanh', 0, 'gat_4', 'linear', 'add'], 'hyper_param': [0.001, 0.8, 0.005, 0.001, 64], 'feature_engine': ['origin']},
            'Pubmed': {'action': [1, 'gat_1', 'leaky_relu', 0, 'gcn', 'leaky_relu', 'add'], 'hyper_param': [0.01, 0.4, 0.0005, 5e-05, 128], 'feature_engine': ['origin']},
            'chameleon':  {'action': [0, 'arma', 'relu', 2, 'arma', 'linear', 'add'], 'hyper_param': [5e-3, 0.6, 5e-05, 5e-04, 256], 'feature_engine': ['origin']},
            'cornell': {'action': [0, 'arma', 'tanh', 0, 'arma', 'tanh', 'add'], 'hyper_param': [0.01, 0.2, 0.0005, 0.001, 512], 'feature_engine': ['origin']},
            'texas': {'action': [1, 'cheb', 'leaky_relu', 2, 'arma', 'leaky_relu', 'max'], 'hyper_param': [0.005, 0.7, 0.0005, 0.0005, 256], 'feature_engine': ['origin']},
            'wisconsin': {'action': [1, 'arma', 'linear', 0, 'gat_4', 'leaky_relu', 'max'], 'hyper_param': [0.01, 0.3, 0.01, 0.0005, 128], 'feature_engine': ['origin']},
            
        }
    }

    structure = structure_list[args.split_type][args.dataset]

    logger.info(f"eval arc: {structure}")

    # train from scratch to get the final score
    test_scores_list = []
    valid_scores_percent = []
    test_acc_percent = []

    for i in range(100):
        structure_copy = copy.deepcopy(structure)
        val_acc, test_acc = evaluate(structure_copy, model_manager, train_epoch=500,
                                        retrain_stage="bst_retrain")
        if test_acc == 0:
            continue

        valid_scores_percent.append(val_acc * 100.0)
        test_scores_list.append(test_acc)
        test_acc_percent.append(test_acc * 100.0)

    test_acc_percent.sort()
    valid_scores_percent.sort()
    
    logger.info(
        f"\nbest results: {structure}\n"
        f"\nbest acc(%): {np.mean(test_acc_percent[5:-5]):.8f}  std(%): {np.std(test_acc_percent[5:-5])} var(%): {np.var(test_acc_percent[5:-5])}"
        f"\nvalid acc(%): {np.mean(valid_scores_percent[5:-5]):.8f}  std(%): {np.std(valid_scores_percent[5:-5])} var(%): {np.var(valid_scores_percent[5:-5])}")

if __name__ == '__main__':
    main()

import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import pandas as pd
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

import gc
import copy
from collections import defaultdict
from sklearn.model_selection import train_test_split, StratifiedKFold

from .micro_gnn import MicroGNN
from data_utils.dataset_all_train import AutoGNNDataset
from data_utils.data_utils_planetoid import format_autograph_data_planetoid
from data_utils.heter_utils import process_heter
from model_utils import EarlyStop, TopAverage, process_action, EarlyStopping, EarlyStoppingLoss
from utils import get_logger
from constant import *
from .micro_search_space import MicroSearchSpace

logger = get_logger()

MILLION = 1e6


def evaluate(output, labels, mask):
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices[mask] == labels[mask])
    return correct.item() * 1.0 / mask.sum().item()


def calc_param_num(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)

    return total_params / MILLION, total_trainable_params / MILLION

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

class MicroModelManager(object):
    def __init__(self, args):
        self.main_args = args
        self.args = {}
        self.early_stop_manager = None
        self.reward_manager = TopAverage(10)

        self.drop_out = IN_DROP
        self.multi_label = MULTI_LABEL
        self.lr = LR
        self.weight_decay = WEIGHT_DECAY

        self.epochs = EPOCHS

        self.train_graph_index = 0
        self.train_set_length = 10

        self.shared_params = None

        self.loss_fn = torch.nn.functional.nll_loss

        self.args["cuda"] = CUDA

        self.args["layers_of_child_model"] = args.num_cells

        logger.info("========" + args.dataset)
        dataset_path = args.dataset

        if args.dataset[0] == '/':
            # Online test
            temp_data_name = args.dataset.split("/")
            dataset_name = temp_data_name[-1]
            dataset = AutoGNNDataset(dataset_path)
            dataset_meta = dataset.get_metadata()
            dataset = dataset.get_data()
            self.dataset = copy.deepcopy(dataset)
            self.dataset_meta = copy.deepcopy(dataset_meta)
            self.data, self.data_features = format_autograph_data_500(dataset, dataset_meta, dataset_name)

        elif args.dataset in ['Cora', 'Citeseer', 'Pubmed']:
            # three datasets with high-homophily
            from collections import Counter

            def process_planetoid():
                if self.main_args.split_type == "supervised":
                    # this type test coressponding to SNAG's settings
                    dataset = Planetoid(root=f'../data/planetoid_data/{dataset_path}',
                                    name=f'{dataset_path}')
                else:
                    # the same setting as all the other setting
                    dataset = Planetoid(root=f'../data/planetoid_data/{dataset_path}',
                                    name=f'{dataset_path}',
                                    transform=T.NormalizeFeatures())
                data = dataset[0]
                edge_file = pd.DataFrame(data.edge_index.cpu().numpy().T, columns=['src_idx', 'dst_idx'])
                edge_file['edge_weight'] = 1.0
                fea_table = pd.DataFrame(data.x.cpu().numpy()).reset_index().rename(columns={'index': 'node_index'})

                if self.main_args.split_type == "supervised":
                    # 622 split the same as SNAG's
                    logger.info('data_split with 622 split')
                    skf = StratifiedKFold(5, shuffle=True)
                    idx = [torch.from_numpy(i) for _, i in skf.split(data.y, data.y)]
                    split = [torch.cat(idx[:1], 0), torch.cat(idx[1:2], 0), torch.cat(idx[2:], 0)]
                    data.train_mask = index_to_mask(split[2], data.num_nodes)
                    data.val_mask = index_to_mask(split[1], data.num_nodes)
                    data.test_mask = index_to_mask(split[0], data.num_nodes)
                    
                    train_indices = [i for i in range(len(data.train_mask)) if data.train_mask[i] == 1]
                    valid_indices = [i for i in range(len(data.val_mask )) if data.val_mask [i] == 1]
                    test_indices = [i for i in range(len(data.test_mask)) if data.test_mask[i] == 1]

                    logger.info(f"train_indices len: {len(train_indices)}")
                    logger.info(f"valid_indices len : {len(valid_indices)}")
                    logger.info(f"test_indices len: {len(test_indices)}")

                    logger.info(f"train_indices: {train_indices}")
                    logger.info(f"valid_indices: {valid_indices}")
                    logger.info(f"test_indices: {test_indices}")

                elif self.main_args.split_type == "standard":
                    # standard split with 20 nodes per class, 500 valid, 1000 test
                    train_mask_numpy, val_mask_numpy, test_mask_numpy = data.train_mask.numpy(), data.val_mask.numpy(), data.test_mask.numpy()  
                    train_indices = [i for i in range(len(train_mask_numpy)) if train_mask_numpy[i] == 1]
                    valid_indices = [i for i in range(len(val_mask_numpy)) if val_mask_numpy[i] == 1]
                    test_indices = [i for i in range(len(test_mask_numpy)) if test_mask_numpy[i] == 1]

                    logger.info(f"train_indices: {len(train_indices)}")
                    logger.info(f"valid_indices: {len(valid_indices)}")
                    logger.info(f"test_indices: {len(test_indices)}")
                
                elif self.main_args.split_type == "full_supervised":
                    # 48% 32% 20% the same as the datasets shown in Geom-GCN's paper which provides 10 splits
                    # to alleviate bias of training
                    splits_file_path = '../data/heter_data/splits/' + args.dataset.lower() + '_split_0.6_0.2_' + str(args.split_id) +'.npz'

                    with np.load(splits_file_path) as splits_file:
                        train_mask = splits_file['train_mask']
                        val_mask = splits_file['val_mask']
                        test_mask = splits_file['test_mask']

                    device = data.train_mask.device
                    data.train_mask = torch.zeros_like(data.train_mask, device=device)
                    data.val_mask = torch.zeros_like(data.val_mask, device=device)
                    data.test_mask = torch.zeros_like(data.test_mask, device=device)

                    train_indices = [i for i in range(len(train_mask)) if train_mask[i] == 1]
                    valid_indices = [i for i in range(len(val_mask)) if val_mask[i] == 1]
                    test_indices = [i for i in range(len(test_mask)) if test_mask[i] == 1]

                    data.train_mask[train_indices] = 1
                    data.val_mask[valid_indices] = 1
                    data.test_mask[test_indices] = 1
                    
                    logger.info(f"train_indices len: {len(train_indices)}")
                    logger.info(f"valid_indices len : {len(valid_indices)}")
                    logger.info(f"test_indices len: {len(test_indices)}")

                    logger.info(f"train_indices: {train_indices}")
                    logger.info(f"valid_indices: {valid_indices}")
                    logger.info(f"test_indices: {test_indices}")
                    
                else:
                    raise Exception("wrong split type")

                train_label = pd.DataFrame(
                    {
                        'node_index': np.where(data.train_mask | data.val_mask)[0],
                        'label': data.y[data.train_mask | data.val_mask].cpu().numpy()})
                test_label = pd.DataFrame(
                    {'node_index': np.where(data.test_mask)[0], 'label': data.y[data.test_mask].cpu().numpy()})
                return {
                        'fea_table': fea_table, 'edge_file': edge_file,
                        'train_indices': train_indices, 'test_indices': test_indices, 'val_indices': valid_indices,
                        'train_label': train_label, 'test_label': test_label
                    }, {'n_class': len(Counter(data.y.cpu().numpy()))}

            dataset, dataset_meta = process_planetoid()
            self.data, self.data_features = format_autograph_data_planetoid(dataset, dataset_meta)

        elif args.dataset in ['chameleon', 'cornell', 'texas', 'wisconsin']:
            # full_supervised for four low-homophily
            dataset_dir_path = '../data/heter_data/'
            spilt_str = '../data/heter_data/splits/' + args.dataset + '_split_0.6_0.2_' + str(args.split_id) +'.npz'
            self.data, self.data_features = process_heter(args.dataset, dataset_dir_path, spilt_str)
        else:
            raise Exception("dataset cannot found")

        self.args["num_class"] = self.n_classes = self.data.y.max().item() + 1
        self.args["edge_num"] = self.data.edge_index.shape[1]
        self.args["num_nodes"] = self.data.num_nodes

        micro_space_temp = MicroSearchSpace()
        operations = copy.deepcopy(micro_space_temp.get_search_space(num_of_nodes=args.num_nodes))

        logger.info(f"operations: {operations}")

        if self.args["edge_num"] >= 1400000:
            operations[1]["value"] = operations[1]["value"][4:]
            operations[3]["value"] = operations[3]["value"][4:]
        self.operations = operations

        logger.info("edge num: {}".format(self.args["edge_num"]))

        self.device = torch.device('cuda' if CUDA else 'cpu')

        self.is_use_early_stop = True if args.use_early_stop == "0" else False

        self.retrain_stage = None

        logger.info(f"is use early stop:? {self.is_use_early_stop}")

    @staticmethod
    def run_model(model, optimizer, loss_fn, data, epochs, early_stop=None, tmp_model_file="geo_citation.pkl",
                half_stop_score=0, retrain_stage=None, return_best=False, cuda=True, need_early_stop=False,
                show_info=False):

        dur = []
        begin_time = time.time()
        best_performance = 0
        min_val_loss = float("inf")
        min_train_loss = float("inf")
        max_val_score = float(0.0)
        model_val_acc = 0

        is_early_stop = False
        stop_epoch = epochs
        for epoch in range(1, epochs + 1):
            model.train()
            t0 = time.time()
            # forward
            logits = model(data.x, data.edge_index)
            logits = F.log_softmax(logits, 1)
            loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

            # evaluate
            model.eval()
            logits = model(data.x, data.edge_index)
            logits = F.log_softmax(logits, 1)
            train_acc = evaluate(logits, data.y, data.train_mask)
            dur.append(time.time() - t0)

            val_acc = evaluate(logits, data.y, data.val_mask)
            test_acc = evaluate(logits, data.y, data.test_mask)

            loss = loss_fn(logits[data.val_mask], data.y[data.val_mask])
            val_loss = loss.item()

            judge_state = val_loss < min_val_loss 

            if judge_state:
                max_val_score = val_acc
                min_val_loss = val_loss
                min_train_loss = train_loss
                model_val_acc = val_acc
                best_performance = test_acc

            if show_info:
                logger.info(
                    "Epoch {:05d} |Train Loss {:.4f} | Vaild Loss {:.4f} | Time(s) {:.4f} | train acc {:.4f} | val_acc {:.4f} | test_acc {:.4f}".format(
                        epoch, train_loss, val_loss, np.mean(dur), train_acc, val_acc, test_acc))

            if early_stop is not None:
                early_stop_method = early_stop.on_epoch_end(epoch, val_loss, train_loss)
                if early_stop_method:
                    is_early_stop = True
                    stop_epoch = epoch
                    break

        return model, model_val_acc, best_performance, stop_epoch

    def build_gnn(self, actions):
        model = MicroGNN(actions, self.in_feats, self.n_classes, layers=self.args["layers_of_child_model"],
                        num_hidden=self.args["num_hidden"],
                        dropout=self.args["in_drop"])
        return model

    def build_data(self, feature_engine_name):
        data = self.data

        data_features = self.data_features

        selected = []
        selected_names = []
        for i in range(len(self.operations[-1]["value"])):
            if self.operations[-1]["value"][i] in feature_engine_name:
                feature_engine = data_features[i]
                selected.append(feature_engine)
                selected_names.append(self.operations[-1]["value"][i])

        x = torch.tensor(pd.concat(selected, axis=1).to_numpy(), dtype=torch.float)
        logger.info(f"feature dim: {x.shape}")
        data.x = x

        del data_features

        return data

    def train(self, actions=None, retrain_stage=None, train_epoch=EPOCHS):

        model_actions = actions['action']
        param = actions['hyper_param']
        self.args["lr"] = param[0]
        self.args["in_drop"] = param[1]
        self.args["weight_decay_gnns"] = param[2]
        self.args["weight_decay_fcs"] = param[3]
        self.args["num_hidden"] = param[4]

        self.retrain_stage = retrain_stage

        # create model
        data = self.build_data(actions["feature_engine"])

        self.args["in_feats"] = self.in_feats = data.num_features

        model = self.build_gnn(model_actions)

        data.to(self.device)

        stop_epoch = 0

        optimizer = None
        try:
            early_stop_manager = None
            if self.args["cuda"]:
                model.cuda()

            optimizer = torch.optim.Adam([{'params':model.params_fc, 'weight_decay': self.args['weight_decay_fcs']},
                                        {'params':model.params_gnn, 'weight_decay': self.args["weight_decay_gnns"]},
                                        ], self.args["lr"])
            if self.is_use_early_stop:
                early_stop_manager = EarlyStoppingLoss(patience=EARLY_STOP_SIZE)
            model, val_acc, test_acc, stop_epoch = self.run_model(model, optimizer, self.loss_fn, data, train_epoch,
                                                                early_stop=early_stop_manager, cuda=self.args["cuda"],
                                                                half_stop_score=max(
                                                                      self.reward_manager.get_top_average() * 0.7, 0.4),
                                                                retrain_stage=retrain_stage)

        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                logger.info(f"we met cuda OOM; error message: {e}")
                val_acc = 0
                test_acc = 0
            else:
                raise e
        
        # destroy model to avoid "cuda OOM"
        del data
        del model
        del optimizer

        torch.cuda.empty_cache()
        gc.collect()

        return val_acc, test_acc, stop_epoch

    def record_action_info(self, origin_action, val_acc, test_acc):
        
        with open(os.path.join(self.main_args.logger_path,
                               self.main_args.search_mode + self.main_args.submanager_log_file),
                  "a") as file:
            # with open(f'{self.args.logger_path}_{self.args.search_mode}_{self.args.format}_manager_result.txt', "a") as file:
            file.write(str(origin_action))

            file.write(";")
            file.write(str(val_acc))

            file.write(";")
            file.write(str(test_acc))

            file.write("\n")

import os
import time

import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

import gc
import copy
from collections import defaultdict
from sklearn.model_selection import train_test_split, StratifiedKFold

from .macro_gnn import MacroGNN
from macro_space.search_space import MacroSearchSpace
from data_utils.dataset_all_train import AutoGNNDataset
from data_utils.autograph_utils import format_autograph_data
from data_utils.data_utils_500 import format_autograph_data_500
from data_utils.data_utils_planetoid import format_autograph_data_planetoid
from model_utils import EarlyStop, TopAverage, process_action, EarlyStopping, EarlyStoppingLoss
from utils import get_logger
from constant import *

logger = get_logger()

MILLION = 1e6


def evaluate(output, labels, mask):
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices[mask] == labels[mask])
    return correct.item() * 1.0 / mask.sum().item()


def calc_param_num(model):
    total_params = sum(p.numel() for p in model.parameters())
    # print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')
    return total_params / MILLION, total_trainable_params / MILLION


class MacroModelManager(object):
    def __init__(self, args):
        self.main_args = args
        self.args = {}
        self.early_stop_manager = None
        self.reward_manager = TopAverage(10)

        self.drop_out = IN_DROP
        self.multi_label = MULTI_LABEL
        self.lr = LR
        self.weight_decay = WEIGHT_DECAY
        # self.retrain_epochs = args.retrain_epochs
        # self.loss_fn = torch.nn.BCELoss()
        self.epochs = EPOCHS
        self.train_graph_index = 0
        self.train_set_length = 10

        # self.param_file = args.param_file
        self.shared_params = None

        self.loss_fn = torch.nn.functional.nll_loss

        self.args["cuda"] = CUDA
        self.args["layers_of_child_model"] = LAYERS_OF_CHILD_MODEL

        logger.info("========" + args.dataset)
        dataset_path = args.dataset
        # dataset = AutoGraphDataset(f"{dataset_path}/train.data", dataset_path)

        if args.dataset[0] == '/':
            temp_data_name = args.dataset.split("/")
            dataset_name = temp_data_name[-1]
            dataset = AutoGNNDataset(dataset_path)
            dataset_meta = dataset.get_metadata()
            dataset = dataset.get_data()
            self.dataset = copy.deepcopy(dataset)
            self.dataset_meta = copy.deepcopy(dataset_meta)
            self.data, self.data_features = format_autograph_data_500(dataset, dataset_meta, dataset_name)

        else:
            from collections import Counter

            def process_planetoid(planetoid):
                data = planetoid[0]
                edge_file = pd.DataFrame(data.edge_index.cpu().numpy().T, columns=['src_idx', 'dst_idx'])
                edge_file['edge_weight'] = 1.0
                fea_table = pd.DataFrame(data.x.cpu().numpy()).reset_index().rename(columns={'index': 'node_index'})

                if self.main_args.split_type == "supervised":
                    # all_nodes_indices = list(range(data.num_nodes))

                    # train_indices, test_indices = train_test_split(all_nodes_indices, test_size=0.2)
                    # train_indices, valid_indices = train_test_split(train_indices, test_size=0.25)

                    # logger.info(f"train_indices: {train_indices}")
                    # logger.info(f"valid_indices: {valid_indices}")
                    # logger.info(f"test_indices: {test_indices}")

                    # # train_indices = all_nodes_indices[:-1000]
                    # # valid_indices = all_nodes_indices[-1000: -500]
                    # # test_indices = all_nodes_indices[-500:]

                    # device = data.train_mask.device
                    # data.train_mask = torch.zeros_like(data.train_mask, device=device)
                    # data.val_mask = torch.zeros_like(data.val_mask, device=device)
                    # data.test_mask = torch.zeros_like(data.test_mask, device=device)
                    # data.train_mask[train_indices] = 1
                    # data.val_mask[valid_indices] = 1
                    # data.test_mask[test_indices] = 1

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

                    train_mask_numpy, val_mask_numpy, test_mask_numpy = data.train_mask.numpy(), data.val_mask.numpy(), data.test_mask.numpy()  
                    train_indices = [i for i in range(len(train_mask_numpy)) if train_mask_numpy[i] == 1]
                    valid_indices = [i for i in range(len(val_mask_numpy)) if val_mask_numpy[i] == 1]
                    test_indices = [i for i in range(len(test_mask_numpy)) if test_mask_numpy[i] == 1]

                    logger.info(f"train_indices: {len(train_indices)}")
                    logger.info(f"valid_indices: {len(valid_indices)}")
                    logger.info(f"test_indices: {len(test_indices)}")
                    
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

            dataset = Planetoid(root=f'../../planetoid_data/{dataset_path}',
                                name=f'{dataset_path}',
                                transform=T.NormalizeFeatures())

            dataset, dataset_meta = process_planetoid(dataset)
            self.data, self.data_features = format_autograph_data_planetoid(dataset, dataset_meta)
            # self.data = dataset[0]
            # self.data_features = [self.data]
            # logger.info(f"self.data:\n {self.data}")

        # self.args["in_feats"] = self.in_feats = self.data.num_features
        self.args["num_class"] = self.n_classes = self.data.y.max().item() + 1
        self.args["edge_num"] = self.data.edge_index.shape[1]
        self.args["num_nodes"] = self.data.num_nodes

        macro_space_temp = MacroSearchSpace()
        operations = copy.deepcopy(macro_space_temp.get_search_space(LAYERS_OF_CHILD_MODEL))
        
        # if self.args["edge_num"] >= 1400000:
        #     operations[1]["value"] = operations[1]["value"][4:]
        #     operations[3]["value"] = operations[3]["value"][4:]
        self.operations = operations

        logger.info("edge num: {}".format(self.args["edge_num"]))

        self.device = torch.device('cuda' if CUDA else 'cpu')
        # self.data.to(self.device)

        self.is_use_early_stop = True if args.use_early_stop == "0" else False

        self.retrain_stage = None

        logger.info(f"is use early stop:? {self.is_use_early_stop}")

    @staticmethod
    def run_model(model, optimizer, loss_fn, data, epochs, early_stop=None, tmp_model_file="geo_citation.pkl",
                  half_stop_score=0, retrain_stage=None, search_space="micro", return_best=False, cuda=True, need_early_stop=False,
                  show_info=False):

        dur = []
        begin_time = time.time()
        best_performance = 0
        min_val_loss = float("inf")
        min_train_loss = float("inf")
        max_val_score = float(0.0)
        model_val_acc = 0
        # print("Number of train datas:", data.train_mask.sum())
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

            # if retrain_stage is not None:
            #     if test_acc > best_performance:
            #         best_performance = test_acc
            
            if search_space == "macro":
                judge_state = val_loss < min_val_loss  # and train_loss < min_train_loss
                # judge_state = (val_acc > max_val_score) or (val_acc == max_val_score and val_loss < min_val_loss)
            else:
                judge_state = (val_acc > max_val_score) or (val_acc == max_val_score and val_loss < min_val_loss)
            if judge_state:
                max_val_score = val_acc
                min_val_loss = val_loss
                min_train_loss = train_loss
                model_val_acc = val_acc
                if test_acc > best_performance:
                    best_performance = test_acc
                # best_performance = test_acc

            if show_info:
                logger.info(
                    "Epoch {:05d} |Train Loss {:.4f} | Vaild Loss {:.4f} | Time(s) {:.4f} | train acc {:.4f} | val_acc {:.4f} | test_acc {:.4f}".format(
                        epoch, train_loss, val_loss, np.mean(dur), train_acc, val_acc, test_acc))

                end_time = time.time()
                # print("Each Epoch Cost Time: %f " % ((end_time - begin_time) / epoch))

            # judge if converge
            # if early_stop is not None:
            #     if early_stop.should_stop(epoch, train_loss, train_acc, val_loss, val_acc):
            #         # logger.info(f"cur epoch: {epoch} but early stop")
            #         is_early_stop = True
            #         stop_epoch = epoch
            #         break
            if early_stop is not None:
                if early_stop.on_epoch_end(epoch, val_acc, train_loss):
                    is_early_stop = True
                    stop_epoch = epoch
                    break

        # logger.info(f"val_score:{model_val_acc}, test_score:{best_performance}")
        # print(f"val_score:{model_val_acc}, test_score:{best_performance}")
        # if return_best:
        #     return model, model_val_acc, best_performance
        # else:
        #     return model, model_val_acc, best_performance

        return model, model_val_acc, best_performance, stop_epoch

    def build_gnn(self, actions):
        model = MacroGNN(actions, self.in_feats, self.n_classes,
                         drop_out=self.args["in_drop"],
                         batch_normal=False, residual=False)
        return model

    def build_data(self, feature_engine_name):
        # feature_engine_name: e.g. ["svd", "edge_weights", "lpa"]
        # data = copy.deepcopy(self.data)
        # return self.data_features[0]
        data = self.data

        # if self.retrain_stage is not None:
        #     # if self.retrain_stage == "revalid":
        #     #     train_indices = copy.deepcopy(self.train_indices)
        #     #     train_indices, val_indices = train_test_split(train_indices, test_size=0.25)
        #     #
        #     #     train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        #     #     train_mask[train_indices] = 1
        #     #     data.train_indices = np.asarray(train_indices)
        #     #     data.train_mask = train_mask
        #     #
        #     #     val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        #     #     val_mask[val_indices] = 1
        #     #     data.val_indices = val_indices
        #     #     data.val_mask = val_mask
        #     #
        #     #     logger.info(
        #     #         f"train num: {len(train_indices)}; valid num: {len(val_indices)}")
        #     #
        #     # elif self.retrain_stage == "bst_retrain":
        #     #     dataset = copy.deepcopy(self.dataset)
        #     #     dataset_meta = copy.deepcopy(self.dataset_meta)
        #     #     data, self.data_features, self.all_train_indices, self.train_indices = format_autograph_data(dataset, dataset_meta)
        #     #
        #     #     del dataset
        #     #     del dataset_meta
        #     #
        #     # else:
        #     #     logger.info("error in retrain_stage name")
        #     dataset = copy.deepcopy(self.dataset)
        #     dataset_meta = copy.deepcopy(self.dataset_meta)
        #     data, self.data_features, self.all_train_indices, self.train_indices = format_autograph_data(dataset,
        #                                                                                                  dataset_meta)
        #     del dataset
        #     del dataset_meta

        # data_features = copy.deepcopy(self.data_features)

        # selected = [self.data_features[0], self.data_features[1]]
        # x = torch.tensor(pd.concat(selected, axis=1).to_numpy(), dtype=torch.float)
        #
        # data.x = x
        #
        # return data

        data_features = self.data_features

        selected = []
        selected_names = []
        for i in range(len(self.operations[-1]["value"])):
            if self.operations[-1]["value"][i] in feature_engine_name:
                feature_engine = data_features[i]
                selected.append(feature_engine)
                selected_names.append(self.operations[-1]["value"][i])

        # logger.info(f"selected_names: {selected_names}")
        # logger.info(f"selected:\n {pd.concat(selected, axis=1)}")

        # origin_x = self.data.x.numpy()
        # fea_x = pd.concat(selected, axis=1).to_numpy()

        # if np.all(origin_x == fea_x):
        #     logger.info("==============equal============")
        # else:
        #     logger.info(f"origin_x shape: {origin_x.shape}")
        #     logger.info(f"fea_x shape: {fea_x.shape}")
        #
        #     logger.info(f"origin_x: {origin_x}")
        #     logger.info(f"fea_x: {fea_x}")

            # assert origin_x.shape == fea_x.shape
            # for i in range(origin_x.shape[0]):
            #     for j in range(origin_x.shape[1]):
            #         if origin_x[i][j] != fea_x[i][j]:
            #             logger.info(f"not equal: {i},{j}: {origin_x[i][j]},{fea_x[i][j]}")

        x = torch.tensor(pd.concat(selected, axis=1).to_numpy(), dtype=torch.float)
        logger.info(f"feature dim: {x.shape}")
        data.x = x

        del data_features

        return data

    def train(self, actions=None, retrain_stage=None, train_epoch=EPOCHS):
        
        model_actions = actions['action']
        model_actions.append(self.args["num_class"])
        param = actions['hyper_param']
        self.args["lr"] = param[0]
        self.args["in_drop"] = param[1]
        self.args["weight_decay"] = param[2]

        # logger.info(f"current arc: {actions}")
        self.retrain_stage = retrain_stage

        # create model
        data = self.build_data(actions["feature_engine"])

        self.args["in_feats"] = self.in_feats = data.num_features

        model = self.build_gnn(model_actions)

        data.to(self.device)

        # if retrain_stage is not None:
        #     total_params, total_trainable_params = calc_param_num(model)

        stop_epoch = 0

        optimizer = None
        try:
            early_stop_manager = None
            if self.args["cuda"]:
                model.cuda()
            # use optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args["lr"], weight_decay=self.args["weight_decay"])
            if self.is_use_early_stop:
                # early_stop_manager = EarlyStop(EARLY_STOP_SIZE)
                early_stop_manager = EarlyStopping(patience=EARLY_STOP_SIZE) if self.main_args.search_space == "micro" else EarlyStoppingLoss(patience=EARLY_STOP_SIZE)
            model, val_acc, test_acc, stop_epoch = self.run_model(model, optimizer, self.loss_fn, data, train_epoch,
                                                                  early_stop=early_stop_manager, cuda=self.args["cuda"],
                                                                  half_stop_score=max(
                                                                      self.reward_manager.get_top_average() * 0.7, 0.4),
                                                                  retrain_stage=retrain_stage, search_space=self.main_args.search_space)
            if retrain_stage is None:
                self.record_action_info(actions, val_acc, test_acc)

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

        # if retrain_stage is not None:
        #     return val_acc, test_acc, stop_epoch, total_params, total_trainable_params
        # else:
        #     return val_acc, test_acc, stop_epoch

    def record_action_info(self, origin_action, val_acc, test_acc):
        # logger.info(
        #     f"WRITEWRITE {os.path.join(self.main_args.logger_path, self.main_args.search_mode + self.main_args.submanager_log_file)}")
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

import copy
import scipy as sp
import numpy as np
import pandas as pd
from sparsesvd import sparsesvd
from scipy.sparse.linalg import svds
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.linear_model import logistic
from sklearn.calibration import CalibratedClassifierCV
from utils import *

def svd_feature(edge_df=None, meta_info=None, num_feature=64, **kwargs):
    num_nodes = meta_info['num_nodes']
    adj_matrix = np.zeros((num_nodes, num_nodes))
    adj_matrix[edge_df['src_idx'], edge_df['dst_idx']] = 1.0
    sparse_adj_matrix = sp.sparse.csc_matrix(adj_matrix)
    # ut, s, vt = sparsesvd(sparse_adj_matrix, num_feature)
    # svd_feats = np.dot(ut.T, np.diag(s))
    u, s, vt = svds(sparse_adj_matrix, num_feature)
    svd_feats = np.dot(u, np.diag(s))
    
    return pd.DataFrame(svd_feats)


def edge_weights_feature(edge_df=None, meta_info=None, **kwargs):
    num_nodes = meta_info['num_nodes']
    edge_weights = np.zeros((num_nodes, num_nodes))
    edge_weights[edge_df['src_idx'], edge_df['dst_idx']] = edge_df['edge_weight']
    return pd.DataFrame(edge_weights)


def normalize_feature(feat_df=None, **kwargs):
    x_feat = feat_df.drop('node_index', axis=1).to_numpy().astype(dtype=np.float64)
    inv_x_rowsum = np.power(x_feat.sum(axis=1), -1).flatten()
    inv_x_rowsum[np.isinf(inv_x_rowsum)] = 0.
    x_diag_mat = np.diag(inv_x_rowsum)
    normalized_x = x_diag_mat.dot(x_feat)
    return pd.DataFrame(normalized_x)


class SVM:
    def __init__(self, **kwargs):
        self.name = "SVM"
        self._model = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=500, class_weight=None, random_state=666))

    def fit(self, x_train, y_train):
        self._model.fit(x_train, y_train)

    def predict(self, x_test):
        return self._model.predict_proba(x_test)


class LR:
    def __init__(self, **kwargs):
        self.name = "LR"
        self._model = logistic.LogisticRegression(C=1.0, solver="liblinear", multi_class="auto",
                                                class_weight=None, max_iter=100, random_state=666)

    def fit(self, x_train, y_train):
        self._model.fit(x_train, y_train)

    def predict(self, x_test):
        return self._model.predict_proba(x_test)


def prepredict_feature(
        feat_df=None, meta_info=None, train_indices=None, val_indices=None, train_label=None, test_indices=None, use_ohe=False,
        **kwargs
):
    fea_table = feat_df.set_index(keys="node_index")
    test_indices = copy.deepcopy(test_indices)
    if val_indices is not None:
        test_indices += val_indices

    if len(train_indices) == 20 * meta_info['num_class']:
        test_indices = list(set(range(0, meta_info['num_nodes'])).difference(set(train_indices)))

    train_label = train_label.set_index('node_index').loc[train_indices][['label']]
    x_train, y_train = fea_table.loc[train_indices].to_numpy(), train_label.to_numpy()
    x_test = fea_table.loc[test_indices].to_numpy()
    lr = LR()
    lr.fit(x_train, y_train)

    if use_ohe:
        ohe = OneHotEncoder(handle_unknown="ignore").fit(y_train.reshape(-1, 1))
        x_train_feat, x_test_feat = ohe.transform(np.argmax(lr.predict(x_train), axis=1).reshape(-1, 1)).toarray(), \
                                    ohe.transform(np.argmax(lr.predict(x_test), axis=1).reshape(-1, 1)).toarray()
    else:
        x_train_feat, x_test_feat = lr.predict(x_train), \
                                    lr.predict(x_test)
    pre_feat = np.concatenate([x_train_feat, x_test_feat], axis=0)
    get_logger().info(f"prepredict: {pre_feat[:20]}")
    total_indices = np.concatenate([train_indices, test_indices], axis=0)
    return pd.DataFrame(data=pre_feat, index=total_indices)


def lpa_feature(
        edge_df=None, meta_info=None,
        train_label=None, train_indices=None, val_indices=None, test_indices=None,
        use_ohe=False, max_iter=100, tol=1e-3,
        **kwargs
):
    test_indices = copy.deepcopy(test_indices)
    if val_indices is not None:
        test_indices += val_indices
    if len(train_indices) == 20 * meta_info['num_class']:
        test_indices = list(set(range(0, meta_info['num_nodes'])).difference(set(train_indices)))

    train_label = train_label.set_index('node_index').loc[train_indices][['label']].to_numpy()
    train_label = train_label.reshape(-1)
    edges = edge_df.to_numpy()
    edge_index = edges[:, :2].astype(np.int).transpose()  # transpose to (2, num_edges)
    edge_weight = edges[:, 2].astype(np.float)
    num_nodes = meta_info['num_nodes']

    total_indices = np.concatenate([train_indices, test_indices], axis=0)
    adj = sp.sparse.coo_matrix((edge_weight, edge_index), shape=(num_nodes, num_nodes)).tocsr()
    adj = adj[total_indices]  # reorder
    adj = adj[:, total_indices]

    row_sum = np.array(adj.sum(axis=1), dtype=np.float)
    d_inv = np.power(row_sum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    normal_adj = sp.sparse.diags(d_inv).dot(adj).tocsr().transpose()

    n_class = meta_info['num_class']
    Pll = normal_adj[:len(train_indices), :len(train_indices)].copy()
    Plu = normal_adj[:len(train_indices), len(train_indices):].copy()
    Pul = normal_adj[len(train_indices):, :len(train_indices)].copy()
    Puu = normal_adj[len(train_indices):, len(train_indices):].copy()
    label_mat = np.eye(n_class)[train_label]
    label_mat_prob = label_mat.copy()
#     print("Pul shape {}, label_mat shape {}".format(Pul.shape, label_mat_prob.shape))

    Pul_dot_lable_mat = Pul.dot(label_mat)
    unlabel_mat = np.zeros(shape=(len(test_indices), n_class))
    iter, changed = 0, np.inf
    while iter < max_iter and changed > tol:
        iter += 1
        pre_unlabel_mat = unlabel_mat
        unlabel_mat = Puu.dot(unlabel_mat) + Pul_dot_lable_mat
        label_mat_prob = Pll.dot(label_mat_prob) + Plu.dot(pre_unlabel_mat)
        changed = np.abs(pre_unlabel_mat - unlabel_mat).sum()
    # preds = np.argmax(np.array(unlabel_mat), axis=1)
    # unlabel_mat = np.eye(n_class)[preds]
    total_indices = np.concatenate([train_indices, test_indices], axis=0)
    if use_ohe:
        ohe = OneHotEncoder(handle_unknown="ignore").fit(train_label.reshape(-1, 1))
        label_mat_ohe = ohe.transform(np.argmax(label_mat_prob, axis=1).reshape(-1, 1)).toarray()
        unlabel_mat_ohe = ohe.transform(np.argmax(unlabel_mat, axis=1).reshape(-1, 1)).toarray()
        lu_mat_ohe = np.concatenate([label_mat_ohe, unlabel_mat_ohe], axis=0)
        return pd.DataFrame(data=lu_mat_ohe, index=total_indices)
    else:
        unlabel_mat_prob = unlabel_mat
        lu_mat_prob = np.concatenate([label_mat_prob, unlabel_mat_prob], axis=0)
        get_logger().info(f"lpa: {lu_mat_prob[-20:]}")
        return pd.DataFrame(data=lu_mat_prob, index=total_indices)


def degree_bins_feature(
        meta_info=None, feat_df=None, edge_df=None, degree_series=None,
        **kwargs
):
    is_undirected = meta_info['is_undirected']
    if is_undirected:
        degree = feat_df['node_index'].map(edge_df.groupby('src_idx').size().to_dict())
        degree.fillna(0, inplace=True)
    else:
        out_degree = feat_df['node_index'].map(edge_df.groupby('src_idx').size().to_dict())
        in_degree = feat_df['node_index'].map(edge_df.groupby('dst_idx').size().to_dict())
        out_degree.fillna(0, inplace=True)
        in_degree.fillna(0, inplace=True)
        degree = in_degree + out_degree
    degree_series = degree
    bins = int(max(30, degree_series.nunique() / 10))
    degree_counts = degree_series.value_counts().reset_index()
    degree_counts = degree_counts.rename(columns={'index': 'degree', 'node_index': 'nums'})
    degree_counts = degree_counts.sort_values('degree')

    min_nums = degree_series.shape[0] / bins
    k = 0
    cum_nums = 0
    bins_dict = {}
    for i, j in zip(degree_counts['degree'], degree_counts['nums']):
        cum_nums += j
        bins_dict[i] = k
        if cum_nums >= min_nums:
            k += 1
            cum_nums = 0

    degree_bins = degree_series.map(bins_dict)
    return pd.concat([degree_series, degree_bins], axis=1)

# TODO: maybe asister's neighbour bin 2

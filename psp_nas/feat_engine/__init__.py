# 分为两步: reduction(预处理) + expansion(构造特征)
#   预处理:
#   - 去掉unique_counts == 1
#   - 判断是否meaningful weight
#   特征工程
#   - svd: SunJunWei, 对adj_matrix进行svd分解
#   - ohe: 给node index进行ohe编码


"""
The interface of all .*_feature

Args:
    meta_info: dict of the task metainfo {'num_nodes': Integer, 'num_class': Integer, ...},
    feat_df: pd.DataFrame['node_index', 'feat_1', ..., 'feat_n'],
    edge_df: pd.DataFrame['src_idx', 'dst_idx', 'edge_weight'],
    train_label: pd.DataFrame['node_index', 'label'],
    train_indices: list of the index of train set,
    test_indices: list of the index of test set (no valid set),
    use_ohe: whether use one hot encoder specially in some feature,
    max_iter: specially in lpa,
    tol: specially in lpa,
    **kwargs: redundant

Returns: pd.DataFrame shape(num_nodes x dim_feature)

"""

__all__ = [
    'drop_n_unique',
    'svd_feature', 'edge_weights_feature', 'normalize_feature',
    'prepredict_feature', 'lpa_feature', 'degree_bins_feature'
]


from .reduction import drop_n_unique
from .expansion import svd_feature, edge_weights_feature, normalize_feature,\
    prepredict_feature, lpa_feature, degree_bins_feature



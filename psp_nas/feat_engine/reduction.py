from utils import get_logger

logger = get_logger()


def drop_n_unique(*args, feat_df=None, n=1, **kwargs):
    unique_counts = feat_df.nunique()
    unique_counts = unique_counts[unique_counts <= n]
    feat_df.drop(unique_counts.index, axis=1, inplace=True)
    return feat_df

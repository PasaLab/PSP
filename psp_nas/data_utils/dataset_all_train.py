# TODO: preprocess dataset for Trainer
import copy
from os.path import join
from datetime import datetime
import numpy as np
import pandas as pd
import yaml

TYPE_MAP = {
    'cat': str,
    'multi-cat': str,
    'str': str,
    'num': np.float64,
    'timestamp': 'str'
}

VERBOSITY_LEVEL = 'WARNING'
TIMESTAMP_TYPE_NAME = 'timestamp'
TRAIN_FILE = 'train_node_id.txt'
TRAIN_LABEL = 'train_label.tsv'
TEST_FILE = 'test_node_id.txt'
INFO_FILE = 'config.yml'
FEA_TABLE = 'feature.tsv'
EDGE_FILE = 'edge.tsv'
SEP = '\t'


def _date_parser(millisecs):
    if np.isnan(float(millisecs)):
        return millisecs

    return datetime.fromtimestamp(float(millisecs))


class AutoGNNDataset:
    """"AutoGNNDataset"""
    def __init__(self, dataset_dir):
        """
            train_indices: np.array
            train_label, fea_table, edge_data: pd.DataFrame
        """
        self.dataset_dir_ = join(dataset_dir, 'train.data')
        self.metadata_ = self._read_metadata(join(self.dataset_dir_, INFO_FILE))
        self.train_indices = None
        self.train_label = None
        self.fea_table = None
        self.edge_data = None

    def get_data(self):
        """get all training data"""
        data = {
            'train_indices': self.get_train_indices(),
            'train_label': self.get_train_label(),
            'fea_table': self.get_fea_table(),
            'edge_file': self.get_edge()
        }
        return data

    def get_metadata(self):
        """get metadata"""
        return copy.deepcopy(self.metadata_)

    def get_fea_table(self):
        """get train"""
        if self.fea_table is None:
            self.fea_table = self._read_dataset(
                join(self.dataset_dir_, FEA_TABLE))
        return self.fea_table

    def get_edge(self):
        """get edge file"""
        dtype = {
            'src_id': int,
            'dst_idx': int,
            'edge_weight': float
        }
        if self.edge_data is None:
            self.edge_data = pd.read_csv(
                join(self.dataset_dir_, EDGE_FILE), dtype=dtype, sep=SEP)
        return self.edge_data

    def get_train_label(self):
        """get train label"""
        dtype = {
            'node_index': int,
            'label': int,
        }
        if self.train_label is None:
            self.train_label = pd.read_csv(
                join(self.dataset_dir_, TRAIN_LABEL), dtype=dtype, sep=SEP)

        return self.train_label

    def get_train_indices(self):
        """get train index file"""
        if self.train_indices is None:
            with open(join(self.dataset_dir_, TRAIN_FILE), 'r') as ftmp:
                self.train_indices = np.array([int(line.strip()) for line in ftmp], dtype=np.int)

        return self.train_indices

    @staticmethod
    def _read_metadata(metadata_path):
        with open(metadata_path, 'r') as ftmp:
            return yaml.safe_load(ftmp)

    def _read_dataset(self, dataset_path):
        schema = self.metadata_['schema']
        if isinstance(schema, dict):
            table_dtype = {key: TYPE_MAP[val] for key, val in schema.items()}
            date_list = [key for key, val in schema.items()
                         if val == TIMESTAMP_TYPE_NAME]
            dataset = pd.read_csv(
                dataset_path, sep=SEP, dtype=table_dtype,
                parse_dates=date_list, date_parser=_date_parser)
        else:
            dataset = pd.read_csv(dataset_path, sep=SEP)

        return dataset


"""Class to load the rs_sar dataset"""
import json
from rs_sar.utils.paths import Paths
import numpy as np


class RSSAR:
    """Class to load rs_sar dataset"""

    def __init__(self):
        self.paths = Paths().get()
        # self.warehouse = self.paths['warehouse']
        self.rs_sar = self.paths['RS-SAR']
        self.data_seq_ref = self._load_data_seq_ref()
        self.annotations = self._load_dataset_ids()
        self.train = dict()
        self.validation = dict()
        self.test = dict()
        self.ratio_spilt = [0.8, 0.1, 0.1]
        self._split()

    def _load_data_seq_ref(self):
        path = self.rs_sar / 'data_seq_ref.json'
        with open(path, 'r', encoding='UTF-8') as fp:
            data_seq_ref = json.load(fp)
        return data_seq_ref

    def _load_dataset_ids(self):
        path = self.rs_sar / 'light_dataset_frame_oriented.json'
        with open(path, 'r', encoding='GBK') as fp:
            annotations = json.load(fp)
        return annotations

    def _split(self):
        for sequence in self.annotations.keys():
            split = self.data_seq_ref[sequence]['split']
            if split == 'Train':
                self.train[sequence] = self.annotations[sequence]
            elif split == 'Validation':
                self.validation[sequence] = self.annotations[sequence]
            elif split == 'Test':
                self.test[sequence] = self.annotations[sequence]
            else:
                raise TypeError('Type {} is not supported for splits.'.format(split))

    def get(self, split):
        """Method to get the corresponding split of the dataset"""
        if split == 'Train':
            return self.train
        if split == 'Validation':
            return self.validation
        if split == 'Test':
            return self.test
        raise TypeError('Type {} is not supported for splits.'.format(split))


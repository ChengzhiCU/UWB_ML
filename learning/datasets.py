from __future__ import print_function
from torch.utils import data
import numpy as np
from config import *
import os

class UWBDataset(data.Dataset):
    def __init__(self, labeled_path, unlabelled_path, train_index_file, is_norm=True):
        self.labeled_path = labeled_path
        self.labeled_data = np.load(self.labeled_path)[()]
        self.is_norm = is_norm

        if len(unlabelled_path) > 0:
            self.unlabeled_path = unlabelled_path
            self.unlabeled_data = np.load(self.unlabeled_data)[()]

        self.index_list = np.load(train_index_file)
        print('index size', self.index_list.shape)

        if is_norm:
            feature = self.labeled_data['extracted_features']
            wave = self.labeled_data['wave']
            self.feature_norm = (feature - np.mean(feature, axis=0)) / np.std(feature, axis=0)
            self.wave_norm = (wave - np.mean(wave, axis=0)) / np.std(wave, axis=0)
            # norm the element of features seperately, as the features have very different mean and std
            # based on the observation of the data, performance is improved by 150%

    def __len__(self):
        return self.index_list.shape[0]

    def __getitem__(self, index):
        data_index = self.index_list[index]
        wave = self.labeled_data['wave'][data_index]
        feature = self.labeled_data['extracted_features'][data_index]
        label = self.labeled_data['label'][data_index]

        label = label - feature[0]
        subject = self.labeled_data['subject'][data_index]

        if self.is_norm:
            feature = self.feature_norm[data_index]
            # feature[0] = 0 #####
            wave = self.wave_norm[data_index]

        return feature, label, subject, wave







from __future__ import print_function
from torch.utils import data
import numpy as np
from config import *
import os


class UWBDataset(data.Dataset):
    def __init__(self, labeled_path, unlabelled_path, train_index_file, enc_type, used_unlabeled=False,
                 regression_delta=True, is_norm=True, wave_cut=True, norm_seperatedly=True):
        self.used_unlabeled = used_unlabeled

        if used_unlabeled == False:
            self.labeled_path = labeled_path
            self.labeled_data = np.load(self.labeled_path)[()]
            self.is_norm = is_norm
            self.regression_delta = regression_delta

            self.index_list = np.load(train_index_file)
            self.labeled_data['mask'] = np.ones_like(self.labeled_data['label'], dtype=np.float32)
            print('index size', self.index_list.shape)

            feature = self.labeled_data['extracted_features']
            self.dis = feature[:, 0]
            wave = self.labeled_data['wave']
            # if wave_cut:
            #     wave = wave[:, 504:]
            #     print('wave shape', wave.shape)
            self.feature_norm = (feature - np.mean(feature, axis=0)) / np.std(feature, axis=0)
            # print(wave.shape, 'mean', np.mean(wave), np.mean(wave).shape, np.mean(np.mean(wave, axis=0), axis=0))
            self.wave_norm = (wave - np.mean(wave)) / np.std(wave)
            # norm the element of features seperately, as the features have very different mean and std
            # based on the observation of the data, performance is improved by 150%
        else:
            self.labeled_path = labeled_path
            self.is_norm = is_norm
            self.regression_delta = regression_delta
            self.labeled_data = {}
            self.unlabeled_path = unlabelled_path

            labeled_data_t = np.load(self.labeled_path)[()]
            unlabeled_data = np.load(self.unlabeled_path)[()]

            labeled_label = labeled_data_t['label']
            labeled_mask = np.ones_like(labeled_label, dtype=np.float32)
            unlab_label = np.zeros_like(unlabeled_data['label'], dtype=np.float32)
            unlabeled_mask = unlab_label

            labeled_wave_len = labeled_label.shape[0]
            unlabeled_wave_len = unlab_label.shape[0]
            labeled_index_list = np.load(train_index_file)
            self.index_list = np.concatenate((labeled_index_list, np.asarray([i for i in range(labeled_wave_len,
                                              labeled_wave_len + unlabeled_wave_len)], dtype=np.int32)), axis=0)
            print('index list shape', self.index_list.shape)

            self.labeled_data['label'] = np.concatenate((labeled_label, unlab_label), axis=0)
            self.labeled_data['mask'] = np.concatenate((labeled_mask, unlabeled_mask), axis=0)

            labeled_feature = labeled_data_t['extracted_features']
            labeled_wave = labeled_data_t['wave']
            # labeled_feature = (labeled_feature - np.mean(labeled_feature, axis=0)) / np.std(labeled_feature, axis=0)
            # print('mean', np.mean(labeled_wave).shape)

            unlab_feature = unlabeled_data['extracted_features']
            unlab_wave = unlabeled_data['wave']
            if norm_seperatedly:
                labeled_wave = (labeled_wave - np.mean(labeled_wave)) / np.std(labeled_wave)
                unlab_wave = (unlab_wave - np.mean(unlab_wave)) / np.std(unlab_wave)

            wave = np.concatenate((labeled_wave, unlab_wave), axis=0)
            feature = np.concatenate((labeled_feature, unlab_feature), axis=0)
            self.dis = feature[:, 0]

            self.labeled_data['wave'] = wave
            self.labeled_data['extracted_features'] = feature
            self.labeled_data['subject'] = np.concatenate((labeled_data_t['subject'],
                                                           np.ones((unlabeled_wave_len, 1))), axis=0)
            # if wave_cut:
            #     wave = wave[:, 504:]
            self.feature_norm = (feature - np.mean(feature, axis=0)) / np.std(feature, axis=0)
            if norm_seperatedly:
                self.wave_norm = wave
            else:
                self.wave_norm = (wave - np.mean(wave)) / np.std(wave)

    def __len__(self):
        return self.index_list.shape[0]

    def __getitem__(self, index):
        data_index = int(self.index_list[index])
        wave = self.labeled_data['wave'][data_index]
        feature = self.labeled_data['extracted_features'][data_index]
        label = self.labeled_data['label'][data_index]
        mask = self.labeled_data['mask'][data_index]
        dis = self.dis[data_index]

        if self.regression_delta:
            label = label - feature[0]
        else:
            pass

        subject = self.labeled_data['subject'][data_index]

        if self.is_norm:
            feature = self.feature_norm[data_index]
            # feature[0] = 0 #####
            wave = self.wave_norm[data_index]

        return feature, label, subject, wave, mask, dis

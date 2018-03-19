from __future__ import print_function
from torch.utils import data
import numpy as np
import config


class UWBDataset(data.Dataset):

    def __init__(self, labeled_path, unlabelled_path, batchsize, mode):
        self.labeled_path = labeled_path
        self.unlabeled_path = unlabelled_path

        self.labeled_data = np.load(self.labeled_data)
        self.unlabeled_data = np.load(self.unlabeled_data)

        labeled_all_data_num = self.labeled_data['label'].shape[0] ####

        try:
            if mode == 'train':
                index_list = np.load('index_list.npz')['train']
            if mode == 'test':
                index_list = np.load('index_list.npz')['test']
        except:

            train_index_num = labeled_all_data_num - config.test_data_num



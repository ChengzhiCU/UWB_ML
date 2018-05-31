import numpy as np
from config import *
import shutil
import scipy.io
from random import shuffle
import random


class ParseMAt:
    def __init__(self, overwrite, input_path=MatDataPath, save_path=PAESED_FILES):
        self.overwrite = overwrite
        self.input_path = input_path
        self.save_path = save_path

    def generate_data_all(self):
        """ generate all the data using multi process"""

        if not os.path.exists(self.save_path):
            print('parse mat files for {}'.format(self.save_path))
            self.make_dir()
            self.generate()

        elif self.overwrite:
            print('Warning! overwrite files exist, all {} processed data will be Deleted! are you sure to continue?'.
                  format("shhs"))
            print('Y/N')
            choice = input().lower()
            if choice != 'y':
                print('please change to False in your code')
                exit(0)
            if os.path.exists(self.save_path):
                shutil.rmtree(self.save_path)
            self.make_dir()
            self.generate()
        else:
            self.make_dir()
            print('generate without overwriting, Warning: Cannot use this method when previous run is stop forcefully')
            self.generate()

    def make_dir(self):
        os.makedirs(self.save_path, exist_ok=True)

    def generate(self):
        filelist = os.listdir(self.input_path)
        L = len(filelist)

        all_list = [i for i in range(len(filelist))]
        # shuffled_list = shuffle(all_list)
        # print('shuffled_list', shuffled_list)
        # train_point_ind = shuffled_list[:int(len(filelist) * 0.8)]
        # test_point_ind = shuffled_list[int(len(filelist) * 0.8):]
        train_point_ind = random.sample(all_list, int(len(filelist) * 0.8))
        train_tr_point_ind = random.sample(train_point_ind, int(len(train_point_ind) * 0.8))
        print(train_tr_point_ind)

        train_tr_data_ind = []
        train_val_data_ind = []
        test_data_ind = []

        num=0
        for i, each in enumerate(filelist):
            matdata = scipy.io.loadmat(os.path.join(self.input_path, each))
            matdata = matdata['data'][0]
            data_array = matdata[0]
            wave_data = data_array[0]
            extracted_feature = data_array[1]
            label = data_array[2]
            scene_index = np.ones_like(label, dtype=np.float32) * i

            block_length = wave_data.shape[0]
            num_list_block = [i for i in range(num, num+block_length)]
            num += block_length
            if i in train_point_ind:
                if i in train_tr_point_ind:
                    train_tr_data_ind = train_tr_data_ind + num_list_block
                else:
                    train_val_data_ind = train_val_data_ind + num_list_block
            else:
                test_data_ind = test_data_ind + num_list_block

            data = {
                'wave': wave_data,
                'extracted_features': extracted_feature,
                'label': label,
                'subject': scene_index
            }
            # np.save(os.path.join(self.save_path, each[:-4]), data)
            print('parse {} succeed'.format(os.path.join(self.save_path, each[:-4])))

            if i==0:
                all_wave = wave_data
                all_extracted_feature = extracted_feature
                all_label = label
                all_subject = scene_index
            else:
                all_wave = np.concatenate((all_wave, wave_data), axis=0)
                all_extracted_feature = np.concatenate((all_extracted_feature, extracted_feature), axis=0)
                all_label = np.concatenate((all_label, label), axis=0)
                all_subject = np.concatenate((all_subject, scene_index), axis=0)

        all_data = {
            'wave': all_wave.astype(np.float32),
            'extracted_features': all_extracted_feature.astype(np.float32),     # dim 5
            'label': all_label.astype(np.float32),
            'subject': all_subject.astype(np.float32)
        }
        np.save(os.path.join(self.save_path, 'all_{}'.format(len(filelist))), all_data)

        data_num = all_label.shape[0]
        # train_data_ind = [i for i in range(data_num) if i%5]
        # test_data_ind = [i for i in range(data_num) if  i%5==0]



        # random
        # train_data_ind = np.random.choice(data_num, int(data_num * 0.8), replace=False)
        # or you can permutate and pick up the first 80% data
        # test_data_ind = [i for i in range(data_num) if not i%5]
        print(test_data_ind[:1000])
        np.save(os.path.join(self.save_path, 'train_tr_ind_sep'), train_tr_data_ind)
        np.save(os.path.join(self.save_path, 'train_val_ind_sep'), train_val_data_ind)
        np.save(os.path.join(self.save_path, 'test_ind_sep'), test_data_ind)







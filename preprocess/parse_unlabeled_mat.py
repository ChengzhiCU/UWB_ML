import numpy as np
from config import *
import shutil
import scipy.io
from random import shuffle
import random


class ParseUnlabeledMAt:
    def __init__(self, overwrite, input_path, save_path):
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

        num=0
        for i, each in enumerate(filelist):
            matdata = scipy.io.loadmat(os.path.join(self.input_path, each))
            matdata = matdata['data'][0]
            data_array = matdata[0]
            wave_data = data_array[0]
            extracted_feature = data_array[1]
            label = data_array[2]
            unlabeled = np.ones_like(label, dtype=np.float32) * (-1)

            # np.save(os.path.join(self.save_path, each[:-4]), data)
            print('parse {} succeed'.format(os.path.join(self.save_path, each[:-4])))

            if i == 0:
                all_wave = wave_data
                all_extracted_feature = extracted_feature
                all_label = unlabeled
                all_subject = unlabeled
            else:
                all_wave = np.concatenate((all_wave, wave_data), axis=0)
                all_extracted_feature = np.concatenate((all_extracted_feature, extracted_feature), axis=0)
                all_label = np.concatenate((all_label, unlabeled), axis=0)
                all_subject = np.concatenate((all_subject, unlabeled), axis=0)

        all_data = {
            'wave': all_wave.astype(np.float32),
            'extracted_features': all_extracted_feature.astype(np.float32),     # dim 5
            'label': all_label.astype(np.float32),
            'subject': all_subject.astype(np.float32)
        }
        np.save(os.path.join(self.save_path, 'unlabeled_{}'.format(len(filelist))), all_data)








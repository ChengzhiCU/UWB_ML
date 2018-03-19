import h5py
import numpy as np
from config import *
import shutil
import scipy

class ParseMAt:

    def __init__(self, overwrite):
        self.overwrite = overwrite
        self.input_path = MatDataPath
        self.save_path = PAESED_FILES

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
        for i, each in enumerate(filelist):
            matdata = scipy.io.loadmat(os.path.join(self.input_path, each))
            matdata = matdata['data'][0]
            data_array = matdata[0]
            wave_data = data_array[0]
            extracted_feature = data_array[1]
            label = data_array[2]
            scene_index = np.ones_like(label, dtype=np.float32) * i
            data = {
                'wave': wave_data,
                'extracted_features': extracted_feature,
                'label': label,
                'subject': scene_index
            }
            np.save(os.path.join(self.save_path, each[:-4]), data)
            print('save {} succeed'.format(os.path.join(self.save_path, each[:-4])))






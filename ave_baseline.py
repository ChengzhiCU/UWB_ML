import numpy as np
import config
import os

data = np.load(os.path.join(config.PAESED_FILES, 'all_12.npy'))[()]
index = np.load(os.path.join(config.PAESED_FILES, 'train_ind.npy'))
import numpy as np
import config
import os

# this is the average baseline for our model

data = np.load(os.path.join(config.PAESED_FILES, 'all_38.npy'))[()]
index = np.load(os.path.join(config.PAESED_FILES, 'train_ind.npy'))

feature = data['extracted_features']
label = data['label']

gap = feature[:, 0] - label[:, 0]

print('no correction', np.mean(np.abs(gap)))
print('bias correction', np.std(gap))

# well, seems our mlp learning 0.1549 outperforms mean method 0.5 a lot
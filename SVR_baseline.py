import numpy as np
import config
import os
from sklearn import svm

# this is the average baseline for our model

regress_error = False

# nlos_folder336 = os.path.join(config.Data_root_path, 'data_block_336_new')
nlos_folder336 = os.path.join(config.Data_root_path, 'los_data_block_new_6f')
data = np.load(os.path.join(nlos_folder336, 'all_146.npy'))[()]
index = np.load(os.path.join(nlos_folder336, 'train_tr_ind_sep.npy'))
index_test = np.load(os.path.join(nlos_folder336, 'test_ind_sep.npy'))
# data = np.load(os.path.join(config.LOS_PAESED_FILES, 'all_88.npy'))[()]
# index = np.load(os.path.join(config.LOS_PAESED_FILES, 'train_tr_ind_sep.npy'))
# index_test = np.load(os.path.join(config.LOS_PAESED_FILES, 'test_ind_sep.npy'))

train_num = index.shape[0]
feature = data['extracted_features']
label = data['label']
train_x = feature[index]
# train_label = np.expand_dims(label[index, 0] - train_x[:, 0], axis=1)
if regress_error:
    train_label = label[index, 0] - train_x[:, 0]
else:
    train_label = label[index, 0]
print(feature.shape[0], train_num, train_x.shape, train_label.shape)

##

test_num = index_test.shape[0]
test_x = feature[index_test]
# test_label = np.expand_dims(test_label[index, 0] - test_x[:, 0], axis=1)
if regress_error:
    test_label = label[index_test, 0] - test_x[:, 0]
else:
    test_label = label[index_test, 0]
print(test_num, test_x.shape, test_label.shape)

# clf = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
# clf = svm.SVR(kernel='poly', C=1e3, degree=2)
# custom kernel http://scikit-learn.org/stable/auto_examples/svm/plot_custom_kernel.html#sphx-glr-auto-examples-svm-plot-custom-kernel-py

clf = svm.SVR()
clf.fit(train_x, train_label)
predict_y = clf.predict(test_x)

rmse_error = (np.sum((predict_y - test_label) ** 2) / test_label.shape[0]) ** 0.5
abs_error = (np.sum(np.abs(predict_y - test_label)) / test_label.shape[0])
print('rmse dis error', rmse_error, 'abs meter error', abs_error)

datasave = {}
datasave['groundtruth'] = test_label
datasave['predict_y'] = predict_y

np.save('temp', datasave)
import scipy.io
scipy.io.savemat(os.path.join(config.MAT_PLOT_PATH, nlos_folder336.split('/')[-1] + '_SVR.mat'), datasave)

from visualization.utils import CDF_plot
CDF_plot(np.abs(predict_y - test_label), 200)


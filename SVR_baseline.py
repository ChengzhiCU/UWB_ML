import numpy as np
import config
import os
from sklearn import svm

# this is the average baseline for our model

data = np.load(os.path.join(config.PAESED_FILES, 'all_38.npy'))[()]
index = np.load(os.path.join(config.PAESED_FILES, 'train_ind.npy'))
train_num = index.shape[0]
feature = data['extracted_features']
label = data['label']
train_x = feature[index]
# train_label = np.expand_dims(label[index, 0] - train_x[:, 0], axis=1)
train_label = label[index, 0] - train_x[:, 0]
print(feature.shape[0], train_num, train_x.shape, train_label.shape)

##
index_test = np.load(os.path.join(config.PAESED_FILES, 'test_ind.npy'))
test_num = index_test.shape[0]
test_x = feature[index_test]
# test_label = np.expand_dims(test_label[index, 0] - test_x[:, 0], axis=1)
test_label = label[index_test, 0] - test_x[:, 0]
print(test_num, test_x.shape, test_label.shape)

# clf = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
# clf = svm.SVR(kernel='poly', C=1e3, degree=2)
clf = svm.SVR()
clf.fit(train_x, train_label)
predict_y = clf.predict(test_x)

error = (np.sum((predict_y - test_label) ** 2) / test_label.shape[0]) ** 0.5
print('dis error', error)
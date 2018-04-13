import numpy as np

datasave = np.load('temp.npy')[()]
test_label = datasave['groundtruth']
predict_y = datasave['predict_y']

blocks_num = 200
import matplotlib.pyplot as plt
gd_max = np.max(test_label)


from visualization.utils import CDF_plot
CDF_plot(np.abs(predict_y - test_label), 200)

#
# pred_abs_error = np.abs(predict_y - test_label)
# pred_error_max = np.max(pred_abs_error)
# pred_error_cnt = np.zeros((blocks_num,))
# step = pred_error_max / blocks_num
#
# for i in range(pred_abs_error.shape[0]):
#     index = int(pred_abs_error[i] / step)-1
#     pred_error_cnt[index] = pred_error_cnt[index] + 1
#
# pred_error_cnt = pred_error_cnt / np.sum(pred_error_cnt)
# CDF = np.zeros((blocks_num,))
#
# for i in range(blocks_num):
#     if i==0:
#         CDF[i] = pred_error_cnt[i]
#     else:
#         CDF[i] = CDF[i-1] + pred_error_cnt[i]
#
# plt.plot(np.linspace(0, pred_error_max, num=blocks_num), CDF)
# plt.show()

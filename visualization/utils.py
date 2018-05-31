import numpy as np
import matplotlib.pyplot as plt


def CDF_plot(data, num, figure_title=""):
    # pred_abs_error = np.abs(predict_y - test_label)
    data = np.abs(data)
    blocks_num = num
    pred_error_max = np.max(data)
    pred_error_cnt = np.zeros((blocks_num + 1,))
    step = pred_error_max / blocks_num

    for i in range(data.shape[0]):
        index = int(data[i] / step)
        pred_error_cnt[index] = pred_error_cnt[index] + 1

    pred_error_cnt = pred_error_cnt / np.sum(pred_error_cnt)
    CDF = np.zeros((blocks_num + 1,))

    for i in range(blocks_num+1):
        if i==0:
            CDF[i] = pred_error_cnt[i]
        else:
            CDF[i] = CDF[i-1] + pred_error_cnt[i]

    plt.plot(np.linspace(0, pred_error_max, num=blocks_num+1), CDF)
    plt.title(figure_title)
    plt.show()
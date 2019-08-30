import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# lstm = np.load('../uav_test/tmpdata/lstmdata.npy')
cnn = np.load('../uav_test/tmpdata/evaluate_cnn.npy')
gtr = np.load('../uav_test/tmpdata/y_test.npy')


def max_iou(y_pred, y_true):
    result = []
    for x, y in zip(y_pred, y_true):
        i = np.sum(np.logical_and(x,y))
        u = np.sum(np.logical_or(x,y))
        result.append(i/u)
    return np.mean(result)

def bn(a):
    for i in range(len(a)):
        a[i] = (a[i] - np.min(a[i])) / (np.max(a[i]) - np.min(a[i]))
    return a


def iocImage():
    # lstmImg = {}
    cnnImg = {}
    # lstmImg['level'] = []
    cnnImg['level'] = []
    # lstmImg['iou'] = []
    cnnImg['iou'] = []

    

    for i in range(5, 10):
        # n=0.5
        # lstm = np.load('../uav_test/tmpdata/lstmdata.npy')
        cnn = np.load('../uav_test/tmpdata/evaluate_cnn.npy')
        gtr = np.load('../uav_test/tmpdata/y_test.npy')
        # bn(lstm)
        bn(cnn)
        bn(gtr)

        n = i / 10
        # lstmImg['level'].append(n)
        cnnImg['level'].append(n)

        # lstm[lstm>=n] = 1
        # lstm[lstm<n]=0
        cnn[cnn>=n] = 1
        cnn[cnn<n] = 0
        gtr[gtr>=n] = 1
        gtr[gtr<n] = 0

        # l = max_iou(lstm, gtr)
        c = max_iou(cnn, gtr)
        # lstmImg['iou'].append(round(l*100, 2))
        cnnImg['iou'].append(round(c*100, 2))
    plt.figure()
    palette = plt.get_cmap('Set1')
    # plt.plot(lstmImg['level'], lstmImg['iou'], color=palette(1), label="lstm")
    plt.plot(cnnImg['level'], cnnImg['iou'], color=palette(2),label="cnn")
    plt.scatter(cnnImg['level'], cnnImg['iou'], color=palette(2),label="cnn")
    plt.xlabel("density level")
    plt.ylabel("IoU(%)")
    plt.title("Intersection over Union")
    plt.legend()
    plt.savefig("img/iou.png")
    

iocImage()

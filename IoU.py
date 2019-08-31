import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# lstm = np.load('../uav_test/tmpdata/lstmdata.npy')
cnn = np.load('tmpdata/evaluate_cnn.npy')
gtr = np.load('tmpdata/y_test.npy')


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
    lstmImg = {}
    lstmImg[0.9] = {}
    lstmImg[0.9]['level'] = []
    lstmImg[0.9]['iou'] = []
    lstmImg[0.8] = {}
    lstmImg[0.8]['level'] = []
    lstmImg[0.8]['iou'] = []
    lstmImg[0.7] = {}
    lstmImg[0.7]['level'] = []
    lstmImg[0.7]['iou'] = []
    cnnImg = {}
    cnnImg[0.9] = {}
    cnnImg[0.9]['level'] = []
    cnnImg[0.9]['iou'] = []
    cnnImg[0.8] = {}
    cnnImg[0.8]['level'] = []
    cnnImg[0.8]['iou'] = []
    cnnImg[0.7] = {}
    cnnImg[0.7]['level'] = []
    cnnImg[0.7]['iou'] = []

    

    for i in [0.9, 0.8, 0.7]:
        # n=0.5
        gtr = np.load('tmpdata/y_test.npy')
        
        bn(gtr)
        gtr[gtr>=i] = 1
        gtr[gtr<i] = 0

        # lstmImg['level'].append(n)
        for j in [0.9, 0.8, 0.7, 0.6, 0.5]:
            lstm = np.load('tmpdata/lstmdata.npy')
            cnn = np.load('tmpdata/evaluate_cnn.npy')
            lstm = bn(lstm)
            cnn = bn(cnn)
            
            # cnnImg[i]['level'].append(j)

            cnn[cnn>=j] = 1
            cnn[cnn<j] = 0
            c = max_iou(cnn, gtr)
            cnnImg[i]['iou'].append(round(c*100, 2))

            lstm[lstm>=j] = 1
            lstm[lstm<j]=0
            l = max_iou(lstm, gtr)
            lstmImg[i]['iou'].append(round(l*100, 2))

    print(cnnImg[0.9]['iou'])

    

    plt.figure(figsize=(12, 4))
    plt.subplots_adjust(wspace=0.5)
    ax = plt.subplot(1, 2, 1)
    palette = plt.get_cmap('Set1')
    # plt.plot(lstmImg['level'], lstmImg['iou'], color=palette(1), label="lstm")
    plt.plot([0.9, 0.8, 0.7, 0.6, 0.5], lstmImg[0.9]['iou'], color=palette(1),label="groundtruth: 0.9-1")
    plt.plot([0.9, 0.8, 0.7, 0.6, 0.5], lstmImg[0.8]['iou'], color=palette(2),label="groundtruth: 0.8-1")
    plt.plot([0.9, 0.8, 0.7, 0.6, 0.5], lstmImg[0.7]['iou'], color=palette(3),label="groundtruth: 0.7-1")
    plt.xlim(0.9, 0.5)
    plt.xlabel("prediction threshold")
    plt.ylabel("IoU(%)")
    plt.title("LSTM Intersection over Union")
    plt.legend()

    ax = plt.subplot(1, 2, 2)
    palette = plt.get_cmap('Set1')
    # plt.plot(lstmImg['level'], lstmImg['iou'], color=palette(1), label="lstm")
    plt.plot([0.9, 0.8, 0.7, 0.6, 0.5], cnnImg[0.9]['iou'], color=palette(1),label="groundtruth: 0.9-1")
    plt.plot([0.9, 0.8, 0.7, 0.6, 0.5], cnnImg[0.8]['iou'], color=palette(2),label="groundtruth: 0.8-1")
    plt.plot([0.9, 0.8, 0.7, 0.6, 0.5], cnnImg[0.7]['iou'], color=palette(3),label="groundtruth: 0.7-1")
    plt.xlim(0.9, 0.5)

    plt.xlabel("prediction threshold")
    plt.ylabel("IoU(%)")
    plt.title("CNN Intersection over Union")
    plt.legend()
    plt.savefig("img/iou.png")
    

iocImage()

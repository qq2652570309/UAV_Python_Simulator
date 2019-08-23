import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


data = [
    # 'data/y_test.npy',
    'data/groundTruths_diff.npy',
    'data/groundTruths_diff.npy',
    'data/groundTruths_diff.npy',
    # 'data/prediction.npy',
    # 'data/positions.npy'
]
rowHeader = ['groundTrue', 'prediction', 'positions']

n = 10
colHeader = ['sample {}'.format(col-1) for col in range(1, n+1)]


plt.figure(figsize=(20, 6))
for index in range(len(data)):
    x = np.load(data[index])
    print(x.shape)
    for i in range(1, n+1):
        # display original
        ax = plt.subplot(3, n, i + index*n)
        ax.set_title(colHeader[i-1])
        if i == 1:
            ax.set_ylabel(rowHeader[index], rotation=90, size='large')
        if rowHeader[index]=='positions':
            if len(x.shape)==4:
                plt.imshow(x[2][i-1], cmap=plt.cm.gnuplot)
            if len(x.shape)==3:
                plt.imshow(x[i-1], cmap=plt.cm.gnuplot)
        else:
            if len(x.shape)==4:
                plt.imshow(x[2][i-1])
            if len(x.shape)==3:
                plt.imshow(x[i-1])
            plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(True)


# plt.show()
plt.savefig("img/density_32_32.png")
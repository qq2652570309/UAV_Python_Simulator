import numpy as np

y = np.load('data/y_test.npy')
p = np.load('data/prediction.npy')
s = np.load('data/positions.npy')

print(y.shape)
print(p.shape)


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

n = 10
rowHeader = ['groundTrue', 'prediction', 'positions']
colHeader = ['sample {}'.format(col-1) for col in range(1, n+1)]


plt.figure(figsize=(20, 6))
for i in range(1, n+1):
    # display original
    ax = plt.subplot(3, n, i)
    ax.set_title(colHeader[i-1])
    if i == 1:
        ax.set_ylabel(rowHeader[0], rotation=90, size='large')
    plt.imshow(y[i-1])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(True)

    # display reconstruction
    ax = plt.subplot(3, n, i + n)
    if i == 1:
        ax.set_ylabel(rowHeader[1], rotation=90, size='large')
    plt.imshow(p[i-1])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(True)

    ax = plt.subplot(3, n, i + 2*n)
    if i == 1:
        ax.set_ylabel(rowHeader[2], rotation=90, size='large')
    plt.imshow(s[i-1], cmap=plt.cm.gnuplot)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(True)

# plt.show()
plt.savefig("img/density_32_32.png")
import numpy as np

y = np.load('data/cnn/y_test.npy')
p = np.load('data/cnn/prediction.npy')

print(y.shape)
print(p.shape)


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

n = 10
rowHeader = ['groundTrue', 'prediction']
colHeader = ['sample {}'.format(col-1) for col in range(1, n+1)]


plt.figure(figsize=(20, 4))
for i in range(1, n+1):
    # display original
    ax = plt.subplot(2, n, i)
    ax.set_title(colHeader[i-1])
    if i == 1:
        ax.set_ylabel(rowHeader[0], rotation=90, size='large')
    plt.imshow(y[i-1])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(True)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    if i == 1:
        ax.set_ylabel(rowHeader[1], rotation=90, size='large')
    plt.imshow(p[i-1])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(True)
# plt.show()
plt.savefig("img/density_with_label.png")

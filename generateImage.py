import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

y_test = np.load('data/groundTruths_density.npy')
prediction = np.load('data/prediction.npy')

y_test = y_test[8500:]
print(y_test.shape)
for i in range(len(y_test)):
    y_test[i] = (y_test[i] - np.min(y_test[i])) / (np.max(y_test[i]) - np.min(y_test[i]))


n = 15
plt.figure(figsize=(30, 4))
for i in range(1, n+1):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(y_test[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(prediction[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
# plt.show()
plt.savefig("img/density_trajectory.png")


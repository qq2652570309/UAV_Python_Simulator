import numpy as np
<<<<<<< HEAD

y = np.load('data/y_test.npy')
p = np.load('data/prediction.npy')

print(y.shape)
print(p.shape)


=======
>>>>>>> pnet_time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

<<<<<<< HEAD
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
plt.savefig("img/density_32_32.png")
=======
class Image:
    def __init__(self, data, rowHeader, colHeader):
        self.data = data
        self.rowHeader = rowHeader
        self.colHeader = ['{0} {1}'.format(colHeader,col-1) for col in range(1, 12)]

    def generate(self):
        n = 11
        plt.figure(figsize=(20, 6))
        for index in range(len(self.data)):
            if '.npy' in self.data[index]:
                x = np.load(self.data[index])
            else:
                x = self.data[index]
                
            print(x.shape)
            if(x.shape[-1]==2):
                x = x[:, :, :, :, 1]
                x.reshape(10, 24, 100, 100)
            for i in range(1, n+1):
                # display original
                ax = plt.subplot(3, n, i + index*n)
                ax.set_title(self.colHeader[i-1])
                if i == 1:
                    ax.set_ylabel(self.rowHeader[index], rotation=90, size='large')
                if self.rowHeader[index]=='positions':
                    if len(x.shape)==4:
                        plt.imshow(x[2][i-1+12], cmap=plt.cm.gnuplot)
                    if len(x.shape)==3:
                        plt.imshow(x[i-1], cmap=plt.cm.gnuplot)
                else:
                    if len(x.shape)==4:
                        plt.imshow(x[2][i-1+12])
                    if len(x.shape)==3:
                        plt.imshow(x[i-1])
                    plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

        plt.savefig("img/analysis.png")


if __name__ == "__main__":
    data = [
        '../../../data/zzhao/uav_regression/cnn/training_data_trajectory.npy',
        '../../../data/zzhao/uav_regression/cnn/training_label_density.npy',
    ]
    rowHeader = ['groundTrue', 'prediction', 'positions']

    i = Image(data, rowHeader, 'test')
    i.generate()
    # print(i.colHeader)
>>>>>>> pnet_time

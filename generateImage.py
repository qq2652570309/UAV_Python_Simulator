import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Image:
    def __init__(self, data, rowHeader, colHeader):
        self.data = data
        self.rowHeader = rowHeader
        self.colHeader = ['{0} {1}'.format(colHeader,col-1) for col in range(1, 11)]

    def generate(self):
        n = 10
        plt.figure(figsize=(20, 6))
        for index in range(len(self.data)):
            if '.npy' in self.data[index]:
                x = np.load(self.data[index])
            else:
                x = self.data[index]
            print(x.shape)
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

        plt.savefig("img/density_cnn_32_32.png")


if __name__ == "__main__":
    data = [
        'data/groundTruths_density.npy',
        'data/groundTruths_density.npy',
        'data/positions.npy',
    ]
    rowHeader = ['groundTrue', 'prediction', 'positions']

    i = Image(data, rowHeader, 'test')
    i.generate()
    # print(i.colHeader)
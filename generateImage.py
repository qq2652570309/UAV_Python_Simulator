import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation

class Image:
    def __init__(self, data, rowHeader, colHeader):
        self.data = data
        self.rowHeader = rowHeader
        self.colHeader = ['{0} {1}'.format(colHeader,col-1) for col in range(1, 11)]

    def load(self, data):
        result = None
        if '.npy' in data:
            result = np.load(data)
        else:
            result = data
        return result

    def generate(self):
        n = 10
        plt.figure(figsize=(20, 6))
        for index in range(len(self.data)):
            x = self.load(self.data[index])
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

    def video(self):
        groundTrue = self.load(data[0])
        prediction = self.load(data[1])
        
        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=1800)

        # First set up the figure, the axis, and the plot element we want to animate
        fig = plt.figure()

        y_test = np.random.rand(5,16,16)
        decoded_imgs = np.random.rand(5,16,16)

        a = np.random.random(groundTrue[0].shape)

        ax1 = plt.subplot(1, 2, 1)
        ax1.set_title('GroundTruth')
        plt.xlim(-1, groundTrue.shape[1])
        plt.ylim(-1, groundTrue.shape[2])
        ax1 = plt.imshow(a, interpolation='none', cmap='binary_r')

        ax2 = plt.subplot(1, 2, 2)
        ax2.set_title('Prediction')
        plt.xlim(-1, groundTrue.shape[1])
        plt.ylim(-1, groundTrue.shape[2])
        ax2 = plt.imshow(a, interpolation='none', cmap='binary_r')

        
        def init():
            ax1.set_data(np.zeros(groundTrue[0].shape))
            ax2.set_data(np.zeros(prediction[0].shape))
            
            
            # ax1.set_data(np.zeros((32, 32)))
            # ax2.set_data(np.zeros((32, 32)))
            return [ax1, ax2]


        def animate(i):
            ax1.set_array(groundTrue[i, :, :])
            ax2.set_array(prediction[i, :, :])
            return [ax1, ax2]


        anim = animation.FuncAnimation(fig, animate, init_func=init,
            frames=groundTrue.shape[0], interval=800, blit=True)
        anim.save('trajectory.mp4', writer=writer)
        

if __name__ == "__main__":
    data = [
        # 'data/groundTruths_density.npy',
        # 'data/groundTruths_density.npy',
        'data/groundTruths_trajctory.npy',
        'data/prediction_trajctory.npy',
        'data/positions.npy',
    ]
    rowHeader = ['groundTrue', 'prediction', 'positions']

    i = Image(data, rowHeader, 'test')
    # i.generate()
    i.video()
    # print(i.colHeader)


import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy

class Correlation:
    def __init__(self):
        # save correlation in data file
        # path = os.path.split(path)[0]
        self.best = 0.0

    def bn(self, a, maxVal, minVal):
        for i in range(len(a)):
            a[i] = (a[i] - minVal) / (maxVal - minVal)
        return a

    def plotmap(self, data):
        plt.figure()
        palette = plt.get_cmap('Set1')
        plt.plot(x, y_pre, color='blue',label="correlation", linewidth=1, zorder=-1)
        plt.scatter(x, y_true, color=palette(0), label="actual", s=0.1, zorder=1)
        plt.xlabel(xlabel)
        plt.ylabel("label")
        plt.title('Correlation Coefficient = {0}'.format(np.round(r,3)))
        plt.legend()
        
        plt.savefig(self.path)
        plt.close()

    def heatmap(self, data, r=0.0, ax=None, xlabel='prediction', cbar_kw={}, cbarlabel="", **kwargs):
        """
        Create a heatmap from a numpy array and two lists of labels.

        Parameters
        ----------
        data
            A 2D numpy array of shape (N, M).
        r
            correlation coefficient
        ax
            A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current axes or create a new one.  Optional.
        cbar_kw
            A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        **kwargs
            All other arguments are forwarded to `imshow`.
        """

        if not ax:
            ax = plt.gca()

        # Plot the heatmap
        im = ax.imshow(data, **kwargs)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
        
        # set label from 0 to 1 and step is 0.1
        x = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        nx = x.shape[0]
        no_labels = 11 # how many labels to see on axis x
        step_x = int(nx / (no_labels - 1)) # step between consecutive labels
        x_positions = np.arange(0,nx,step_x) # pixel count at label position
        x_labels = x[::step_x] # labels you want to see
        plt.xticks([0,10,20,30,40,50,60,70,80,90,100], x_labels)

        y = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        y = y[::-1]
        ny = y.shape[0]
        no_labels = 11 # how many labels to see on axis x
        step_y = int(ny / (no_labels - 1)) # step between consecutive labels
        y_positions = np.arange(0,ny,step_y) # pixel count at label position
        y_labels = y[::step_y] # labels you want to see
        # plt.yticks(y_positions, y_labels)
        plt.yticks([0,10,20,30,40,50,60,70,80,90,100], y_labels)
        
        ax.set_title('Correlation Coefficient = {0}'.format(np.round(r,3)))
        plt.xlabel(xlabel)
        plt.ylabel("label")
        plt.savefig(self.path)
        plt.close()

    def corrcoef(self, prediction, label, path = '.', name = "correlation.png", xlabel="prediction"):
        self.path = os.path.join(path, name)

        predictiond = deepcopy(prediction)
        labeld = deepcopy(label)
        
        # if xlabel == "prediction" :
        #     self.first5Img(matrixImg0=predictiond, matrixImg1=labeld)

        
        
        x = predictiond.reshape((-1))
        y_true = labeld.reshape((-1))
        r = np.corrcoef(x, y_true)
        r = r[0,1]
        
        if r > self.best:
            self.best = r
        r = self.best
        y_pre = x*r

        
        maxVal = 0.0
        maxVal = max(maxVal, np.max(x))
        maxVal = max(maxVal, np.max(y_pre))
        maxVal = max(maxVal, np.max(y_true))
        minVal = 0.0
        minVal = min(minVal, np.min(x))
        minVal = min(minVal, np.min(y_pre))
        minVal = min(minVal, np.min(y_true))
        self.bn(x, maxVal, minVal)
        self.bn(y_pre, maxVal, minVal)
        self.bn(y_true, maxVal, minVal)

        
        grid = np.zeros((101,101))
        for x, y in zip(x, y_true):
            x = round(x*100)
            y = round(y*100)
            grid[int(x),int(y)] += 1
        grid = np.flip(grid, 0)

        # template change, please delete later
        if xlabel=="initial state":
            grid[100,0] = np.partition(grid.flatten(), -2)[-2] * 1.2
        

        grid = grid.reshape(-1)
        self.bn(grid, np.max(grid), np.min(grid))

        # second normalization
        sortedValues = np.unique(grid)
        for i in range(len(grid)) :
            if grid[i] > 0 :
                index, = np.where(np.isclose(sortedValues, grid[i]))
                grid[i] = 0.1 * index[0] 
        
        grid = grid.reshape(101,101)
        
        fig, ax = plt.subplots()
        # self.heatmap(grid, r=r, ax=ax, cmap="YlGn", cbarlabel="frequency")
        self.heatmap(grid, r=r, ax=ax, xlabel=xlabel, cmap="magma_r", cbarlabel="frequency")
        
        return r
        
        

if __name__ == '__main__':
    # prediction = np.load('pData/init.npy')
    # labelp = np.load('pData/label.npy')
    prediction = np.load('../../../data/zzhao/uav_regression/javaP2P/fushion/data_init.npy')
    label = np.load('../../../data/zzhao/uav_regression/javaP2P/fushion/label_mainnet.npy')

    prediction[prediction>10] = 10
    label[label>10] = 10
    
    print(np.unique(label))

    # c = Correlation()
    # c.corrcoef(prediction, label, xlabel="initial state")



    
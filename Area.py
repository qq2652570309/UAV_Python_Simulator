import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

launchingArea = [
    [
        [1, 1],
        [3, 3],
    ],
    [
        [28, 28],
        [30, 30],
    ],
    [
        [23, 23],
        [25, 25],
    ],
]

destinationArea = [
    [
        [3, 26],
        [5, 28],
    ],
    [
        [15, 8],
        [17, 10],
    ],
    [
        [28, 3],
        [30, 5],
    ]
]

blockArea = [
    [15, 5],
    [16, 4],
    [8, 8],
    [8, 12],
    [25, 27],
    [23, 27],
    [15, 15],
    [13, 15],
    [5, 26],
    [18, 6],
]

class Area:
    def __init__(self):
        self.la = np.concatenate(launchingArea, axis=0)
        self.da = np.concatenate(destinationArea, axis=0)
        self.ba = blockArea

    def getLaunchPoint(self, n, low=0, high=1):
        np.random.shuffle(self.la)
        if n == 'random':
            n = int(np.round(np.random.uniform(0, len(self.la))))
        if n == 'all':
            n = len(self.la)
        result = []
        for n in range(n):
            point = random.choice(self.la)
            point = np.append(point, np.random.uniform(low, high))
            result.append(np.round(point, decimals=2))
        return np.array(result)
    
    def getDestination(self, n):
        np.random.shuffle(self.da)
        if n == 'all':
            return self.da
        else:
            point = random.choice(self.da)
            return point[0], point[1]
    
    def getBlockPoint(self, n):
        np.random.shuffle(self.ba)
        if n == 'random':
            n = int(np.round(np.random.uniform(0, len(self.ba))))
        if n == 'all':
            return self.ba
        result = []
        for n in range(n):
            point = random.choice(self.ba)
            result.append(point)
        return np.array(result)
    
    def image(self, size, save=False):
        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        plt.gray()
        for ax, title, area in zip(axs, ['launch', 'destination', 'block'], [self.la, self.da, self.ba]):
            A = np.zeros((size,size))
            for p in area:
                A[p[0], p[1]] = 1
            ax.imshow(A)
            ax.set_title(title)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(True)
        if save == True:
            plt.savefig("img/result.png")

a = Area()
# lp = a.getLaunchPoint('random')
# dp = a.getDestination('random')
# bp = a.getBlockPoint('random')

a.image(32)
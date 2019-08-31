import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

launchingArea = [
    [
        [1, 1],
        [6, 6],
    ],
    [
        [23, 28],
        [28, 33],
    ],
    [
        [48, 18],
        [53, 23],
    ],
    [
        [63, 63],
        [68, 68],
    ],
    [
        [89, 73],
        [94, 78],
    ],
]

destinationArea = [
    [
        [3, 26],
        [8, 31],
    ],
    [
        [14, 55],
        [19, 60],
    ],
    [
        [35, 8],
        [40, 13],
    ],
    [
        [68, 83],
        [73, 88],
    ],
    [
        [88, 3],
        [93, 8],
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
        # self.la = np.concatenate(launchingArea, axis=0)
        # self.da = np.concatenate(destinationArea, axis=0)
        # self.ba = blockArea
        self.la = self.createArea(launchingArea)
        self.da = self.createArea(destinationArea)
        self.ba = blockArea

    def createArea(self, vertices):
        result = []
        vertices = random.sample(vertices, k=3)
        for vertex in vertices:
            start = vertex[0]
            end = vertex[1]
            for row in range(start[0],end[0]+1):
                for col in range(start[1],end[1]+1):
                    result.append([row,col])
        return result

    def getLaunchPoint(self, low=0, high=1, n=None):
        result = []
        possiblity = 0
        for i in range(len(self.la)):
            if i % 25 == 0:
                possiblity = np.random.uniform(low, high)
                # print(self.la[i])
            point = np.append(self.la[i], possiblity)
            result.append(np.round(point, decimals=2))
        if n != None:
            result = random.sample(result, k=n)
        return np.random.permutation(result)
    
    def getDestination(self, allPoints=False):
        if allPoints:
            return self.da
        result = random.choice(np.random.permutation(self.da))
        return result[0], result[1]
    
    def getBlockPoint(self):
        return np.random.permutation(self.ba)
        
    
    def image(self, size, save=False):
        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        plt.gray()
        la = self.getLaunchPoint()
        la = la[:,:2]
        da = self.getDestination(allPoints=True)
        ba = self.getBlockPoint()
        for ax, title, area in zip(axs, ['launch', 'destination', 'block'], [la, da, ba]):
            A = np.zeros((size,size))
            for p in area:
                A[int(p[0]), int(p[1])] = 1
            ax.imshow(A)
            ax.set_title(title)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(True)
        if save == True:
            plt.savefig("img/result.png")

if __name__ == "__main__":
    a = Area()
    a.image(100,save=True)

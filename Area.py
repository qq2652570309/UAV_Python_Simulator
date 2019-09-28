import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    def __init__(self, low, high):
        # self.la = np.concatenate(launchingArea, axis=0)
        # self.da = np.concatenate(destinationArea, axis=0)
        # self.ba = blockArea
        self.refresh()
        self.ba = blockArea

    def refresh(self, mapSize=100, areaSize=3, num=10):
        la, da = self.randomArea(mapSize, areaSize, num)
        possiblity = 0
        for i in range(len(la)):
            if i % 9 == 0:
                possiblity = np.random.uniform()
            la[i] = np.append(la[i], possiblity)
            la[i] = np.round(la[i], decimals=2)
        self.la = la
        self.da = da

    def getLaunchPoint(self, n=None):
        result = self.la
        if n != None:
            result = random.sample(result, k=n)
        return np.array(result)
        # return np.random.permutation(result)
    
    def getDestination(self, allPoints=False):
        if allPoints:
            return np.array(self.da)
        result = random.choice(np.random.permutation(self.da))
        return result[0], result[1]
    
    def getBlockPoint(self):
        return np.random.permutation(self.ba)
    
    def randomArea(self, mapSize=100, areaSize=3, num=10):
        size = mapSize
        locations = []
        for x in range(1,mapSize-areaSize):
            for y in range(1,mapSize-areaSize):
                locations.append((x,y))
        indices = list(range((mapSize-2*areaSize)*(mapSize-2*areaSize)))
        random.shuffle(indices)

        map = np.zeros((mapSize,mapSize))
        areas = []
        select = []

        for i in indices:
            if len(select) is num:
                break
            x ,y = locations[i]
            qualified = True
            for s in select:
                sx ,sy = locations[s]
                if abs(x-sx) < 10 and abs(y-sy) < 10:
                    qualified = False
                    break
            if qualified:
                select.append(i)
                for j in range(0,areaSize):
                    for k in range(0,areaSize):
                        areas.append([x+j,y+k])
                        map[x+j,y+k] = 1

        half = int(len(areas)/2)
        return areas[:half], areas[half:]

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
            plt.savefig("img/AreaMap.png")

if __name__ == "__main__":
    a = Area(0, 1)
    print(a.getLaunchPoint())
    print(a.getDestination(allPoints=True))
    a.image(100,save=True)

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

random.seed(0)
np.random.seed(0)

class Area:
    def __init__(self, mapSize=100, areaSize=3, areaNum=10):
        self.mapSize = mapSize
        self.areaSize = areaSize
        self.areaNum = areaNum
        self.fushion_data_env = np.zeros((3000, 100, 100))
        self.subnet_data_env = np.zeros((180000, 100, 100))
        self.refresh(0)

    def refresh(self, batch):
        self.locations = []
        mapSize = self.mapSize
        areaSize = self.areaSize
        for x in range(1,mapSize-areaSize):
            for y in range(1,mapSize-areaSize):
                self.locations.append((x,y))
        self.indices = list(range((mapSize-2*areaSize)*(mapSize-2*areaSize)))
        random.shuffle(self.indices)
        
        # init no-fly zone
        self.noFlyZone()
        self.fushion_data_env[batch] = self.drawNoFlyZone()
        self.subnet_data_env[batch*60:(batch+1)*60] = self.fushion_data_env[batch]
        
        la, da = self.randomArea(mapSize, areaSize, self.areaNum)
        possiblity = 0
        for i in range(len(la)):
            if i % 9 == 0:
                possiblity = np.random.uniform()
            la[i] = np.append(la[i], possiblity)
            la[i] = np.round(la[i], decimals=2)
        self.la = la
        self.da = da
        
    def drawNoFlyZone(self):
        map = np.zeros((100,100))
        x1, y1 = self.nfz[0]
        x3, y3 = self.nfz[2]
        map[x1:x3+1, y1:y3+1] = -1
        return map
    
    def noFlyZone(self):
        size = random.randint(10,40)
        x1 = 0
        y1 = 0
        while True:
            if 2 <= x1 <= 87 and 2 <= y1 <= 87:
                break
            i = random.choice(self.indices)
            x1, y1 = self.locations[i]
        x2 = x1
        y2 = y1 + size if y1 + size < 100 else 97
        x4 = x1 + size if x1 + size < 100 else 97
        y4 = y1
        x3, y3 = x4, y2
        
        self.nfz = np.array([
            [x1, y1],
            [x2, y2],
            [x3, y3],
            [x4, y4]
        ], dtype=int)

    def updateLaunchingRate(self):
        la = self.la
        possiblity = 0
        for i in range(len(la)):
            if i % 9 == 0:
                possiblity = np.random.uniform()
            la[i][-1] = possiblity
            la[i] = np.round(la[i], decimals=2)
        self.la = la
    
    def getLaunchPoint(self, n=None):
        result = self.la
        if n != None:
            result = random.sample(result, k=n)
        return np.array(result)
    
    def getDestination(self, allPoints=False):
        if allPoints:
            return np.array(self.da)
        result = random.choice(np.random.permutation(self.da))
        return int(result[0]), int(result[1])
    
    def getNoFlyZone(self):
        return self.nfz
    
    def isQualified(self, x, y, select):
        for s in select:
            # is this area too close other areas
            sx ,sy = self.locations[s]
            if abs(x-sx) < 7 and abs(y-sy) < 7:
                return False
            # is this area in no-fly zone
            x1, y1 = self.nfz[0]
            x3, y3 = self.nfz[2]
            if (x1 <= x <= x3 and y1 <= y <= y3) or\
               (x1 <= x <= x3 and y1 <= y+5 <= y3) or\
               (x1 <= x <= x3 and y1 <= y+5 <= y3) or\
               (x1 <= x+5 <= x3 and y1 <= y+5 <= y3):
                return False
        return True
    
    def randomArea(self, mapSize=100, areaSize=3, num=10):
        areas = []
        select = []

        for i in self.indices:
            if len(select) is num:
                break
            x ,y = self.locations[i]
            
            if self.isQualified(x, y, select):
                select.append(i)
                for j in range(0,areaSize):
                    for k in range(0,areaSize):
                        areas.append([x+j,y+k])

        half = int(len(areas)/2)
        return areas[:half], areas[half:]

    def image(self, size, name="test"):
        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        plt.gray()
        la = self.getLaunchPoint()
        la = la[:,:2]
        da = self.getDestination(allPoints=True)
        ba = self.getNoFlyZone()

        for ax, title, area in zip(axs, ['launch', 'destination', 'no-fly zone'], [la, da, ba]):
        # for ax, title, area in zip(axs, ['launch', 'destination', 'block'], [la, da]):
            A = np.zeros((size,size))
            for p in area:
                A[int(p[0]), int(p[1])] = 1
            if title == 'no-fly zone':
                x1, y1 = self.nfz[0]
                x3, y3 = self.nfz[2]
                A[x1:x3+1, y1:y3+1] = 1
            ax.imshow(A)
            ax.set_title(title)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(True)
            plt.savefig("img/{0}.png".format(name))

    def oneImage(self, size, name="test"):
        A = np.zeros((size,size))
        startPositions = self.getLaunchPoint()
        endPositions = self.getDestination(allPoints=True)

        for startRow, startCol, _ in startPositions:
            startRow, startCol = int(startRow), int(startCol)
            A[startRow, startCol] = 1 # purple is 1

        for endRow, endCol in endPositions:
            endRow, endCol = int(endRow), int(endCol)
            A[endRow, endCol] = 2 # red is 2

        ba = self.getNoFlyZone()
        xr = np.arange(ba[0, 0], ba[1, 1])
        yc = np.arange(ba[0, 1], ba[2, 1])
        for x in xr:
            for y in yc:
                A[x,y] = 3 # yellow

        plt.gray()
        frame1 = plt.gca()
        plt.imshow(A, cmap=plt.cm.gnuplot)
        # plt.colorbar()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)

        plt.savefig("img/{0}.png".format(name))


if __name__ == "__main__":
    a = Area(mapSize=100, areaSize=3, areaNum=10)
    
    a.refresh(1)
    a.image(size=100, name="area1")
    a.refresh(2)
    a.image(size=100, name="area2")
    a.refresh(3)
    a.image(size=100, name="area3")
    a.refresh(4)
    a.image(size=100, name="area4")
    # a.oneImage(size=100, name="area4")
    # a.refresh(5)
    # a.image(size=100, name="area5")
    
    
    # for i in range(6):
    #     m = np.all(a.subnet_data_env[i*60:(i+1)*60] == a.subnet_data_env[i*60])
    #     n = np.all(a.subnet_data_env[i*60:(i+1)*60] == a.fushion_data_env[i])
    #     print(m and n, m, n)
    # for i in range(5):
    #     n = np.all(a.fushion_data_env[i] == a.fushion_data_env[i+1])
    #     print(n)
    
    # plt.gray()
    # plt.imshow(a.fushion_data_env[0])
    # plt.savefig("image0.png")
    # plt.imshow(a.fushion_data_env[1])
    # plt.savefig("image1.png")
    # plt.imshow(a.fushion_data_env[5])
    # plt.savefig("image5.png")
    
    
    
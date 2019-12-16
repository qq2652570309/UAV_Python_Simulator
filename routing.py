import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Area import Area

random.seed(0)
np.random.seed(0)

# A = np.zeros((50,10,10))

# A[:, 2:8, 2:8] = -1

# r1,c1 = 2, 2
# r2,c2 = 2, 8
# r3,c3 = 8, 8
# r4,c4 = 8, 2

# sr,sc = 3, 1
# er,ec = 6, 9
# sr,sc = 6, 9
# er,ec = 3, 1

# sr,sc = 5, 0
# er,ec = 4, 8
# sr,sc = 4, 8
# er,ec = 5, 0

# sr,sc = 1, 7
# er,ec = 8, 3
# sr,sc = 8, 3
# er,ec = 1, 7

# A[:,sr,sc] = 2
# A[:,er,ec] = 2


class Routing:
    def __init__(self):
        self.area = Area(mapSize=100, areaSize=3, areaNum=10)
        self.nfz = self.area.getNoFlyZone()
        # self.startValue = 0.25
        # self.endValue = 0.75
        self.startValue = 1
        self.endValue = 2
        self.time = 100

    def routing(self, sr, sc, er, ec, currentTime, trajectors):
        remainingTime = self.time - currentTime
        r1,c1 = self.nfz[0]
        r3,c3 = self.nfz[2]
        
        # Does vertical movement pass no-fly zone?
        if c1 <= sc <= c3 and c1 <= ec <= c3:
            path, pathLen = self.verticalRouting(sr, sc, er, ec)
            self.nfzRouting(path, pathLen, remainingTime, trajectors)
            return
        # Does horizontal movement pass no-fly zone?
        if r1 <= sr <= r3 and r1 <= er <= r3:
            path, pathLen = self.horizontalRouting(sr, sc, er, ec)
            self.nfzRouting(path, pathLen, remainingTime, trajectors)
            return
    
    def nfzRouting(self, path, pathLen, remainingTime, trajectors):
        totalLen = sum(pathLen)
        step = (self.endValue-self.startValue)/(totalLen-1)
        steps = np.around(np.arange(self.startValue, self.endValue+step, step), 2)
        low = 0
        for p, l in zip(path, pathLen):
            if low >= remainingTime:
                break
            
            high = min(low+l,remainingTime)
            values = np.zeros((l))
            values[:high-low] = steps[low:high]
            low += l
            
            rowPath, colPath = p
            ##################### time arange not complete ################
            trajectors[rowPath, colPath] = values
            # trajectors[rowPath, colPath] = 1


    # if horizontal movement will pass no-fly zone, do routing
    def horizontalRouting(self, sr, sc, er, ec):
        R1 = self.nfz[0][0]
        R2 = self.nfz[2][0]
        
        upLen = abs(sr - R1) + abs(er - R1)
        downLen = abs(sr - R2) + abs(er - R2)
        
        lowCol, highCol = min(sc, ec), max(sc, ec)
        orderCol = 1 if sc < ec else -1
        
        path = []
        pathLen = []
        if upLen < downLen:
            path = [
                [np.arange(sr, R1-1, -1), sc],
                [R1-1, np.arange(lowCol, highCol+1)[::orderCol]],
                [np.arange(R1, er+1), ec]
            ]
            pathLen = [abs(sr-R1)+1, abs(lowCol-highCol)+1, +abs(er-R1)+1]
        else:
            path = [
                [np.arange(sr, R2+1), sc],
                [R2+1, np.arange(lowCol, highCol+1)[::orderCol]],
                [np.arange(R2, er-1, -1), ec]
            ]
            pathLen = [abs(sr-R2)+1, abs(lowCol-highCol)+1, +abs(er-R2)+1]
        return path, pathLen

    def verticalRouting(self, sr, sc, er, ec):
        C1 = self.nfz[0][1]
        C2 = self.nfz[2][1]
        
        leftLen = abs(sc - C1) + abs(ec - C1)
        rightLen = abs(sc - C2) + abs(ec - C2)
        
        lowRow, highRow = min(sr, er), max(sr, er)
        orderRow = 1 if sr < er else -1

        path = []
        pathLen = []
        if leftLen < rightLen:
            path = [
                [sr, np.arange(sc, C1-1, -1)],
                [np.arange(lowRow, highRow+1)[::orderRow], C1-1],
                [er, np.arange(C1, ec+1)]
            ]
            pathLen = [abs(sc-C1)+1, abs(lowRow-highRow)+1, abs(ec-C1)+1]
        else:
            path = [
                [sr, np.arange(sc, C2+1)],
                [np.arange(lowRow, highRow+1)[::orderRow], C2],
                [er, np.arange(C2, ec-1, -1)]
            ]
            pathLen = [abs(sc-C2)+1, abs(lowRow-highRow)+1, abs(ec-C2)+1]
        return path, pathLen

    def oneImage(self, size, name="test"):
        trajectors = np.zeros((size,size))
        # sr, sc = 50, 20
        # er, ec = 60, 80
        sr, sc = 70, 20
        er, ec = 60, 80
        
        self.nfz = [
            [45,45],
            [45,75],
            [75,75],
            [75,45]
        ]

        self.routing(sr,sc,er,ec,0,trajectors)
        
        r1,c1 = self.nfz[0]
        r3,c3 = self.nfz[2]
        trajectors[r1:r3, c1:c3] = 1

        plt.gray()
        frame1 = plt.gca()
        plt.imshow(trajectors, cmap=plt.cm.gnuplot)
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)

        plt.savefig("img/{0}.png".format(name))


# path, pathLen = horizontalRouting()
# path, pathLen = verticalRouting()
# for p in path:
#     rowPath, colPath = p
#     A[rowPath, colPath] += 1

r = Routing()
r.oneImage(100, 'routing')




'''
# In channel, 0th is status that uav is launching at this second
# 1st is launching rate of this point
# 2nd and 3rd is (x, y) postion of destination point
'''
import time
import random
import logging
import numpy as np

class Simulator:
    def __init__(self, iteration = 1, time=60, row=0, column=0,  startPointsNum=10, endPointsNum=10):
        self.iteration = iteration
        self.row = row
        self.column = column
        self.time = time
        self.startPointsNum = startPointsNum
        self.endPointsNum = endPointsNum
        # In channel, 0th is status that uav is launching at this second
        # 1st is launching rate of this point
        # 2nd and 3rd is (x, y) postion of destination point
        self.trainingSets = np.zeros(shape=(self.iteration, self.time, self.row, self.column, 4), dtype=np.float32)
        self.groundTruths = np.zeros(shape=(self.iteration, self.time, self.row, self.column), dtype=np.float32)
        logging.info('finish init\n')


    def generate(self):
        startTimeTotal = time.time()
        startPositions, endPositions = self.generateLaunchingPoints()
        
        for index in range(self.iteration):
            startTimeIter = time.time()
            logging.info('At {0} iteration'.format(index))

            for startRow, startCol, launchingRate in startPositions:
                logging.info('   At start Point ({0}, {1})'.format(startRow, startCol))
                # set traning sets
                startRow = int(startRow)
                startCol = int(startCol)
                self.trainingSets[index,:-1,startRow,startCol,1] = launchingRate
                # generate ground truth
                for currentTime in range(self.time):

                    succ = np.random.uniform(0,1) <= self.trainingSets[index,currentTime,startRow,startCol,1]
                    if succ:
                        endRow, endCol  = random.choice(endPositions)
                        remainingTime = self.time - currentTime
                        
                        # add info into channel
                        self.trainingSets[index,currentTime,startRow,startCol,0] = 1 # launching one uav
                        self.trainingSets[index,currentTime,startRow,startCol,2] = endRow # destination row value
                        self.trainingSets[index,currentTime,startRow,startCol,3] = endCol # destination col value
                        
                        logging.info('      At time {0}, ({1}, {2}) --> ({3}, {4})'.format(currentTime, startRow, startCol, endRow, endCol))
                        if remainingTime >= abs(startCol-endCol)+1 :
                            # enough time for horizontal
                            if startCol < endCol :
                                r =  np.arange(startCol, endCol+1)
                            else:
                                r = np.arange(endCol, startCol+1)[::-1]
                        else:
                            # not enough time for horizontal
                            if startCol < endCol:
                                r = np.arange(startCol, startCol+remainingTime)
                            else:
                                r = np.arange(startCol-remainingTime+1, startCol+1)[::-1]
                        t1 = np.arange(currentTime, currentTime+len(r))
                        self.groundTruths[index,t1,startRow,r] += 1
                        remainingTime -= len(r)

                        if remainingTime > 0 :
                            # exists time for vertical
                            if remainingTime >= abs(startRow-endRow) :
                                # enough time for vertical
                                if startRow < endRow:
                                    c = np.arange(startRow+1, endRow+1)
                                else:
                                    c = np.arange(endRow, startRow)[::-1]
                            else:
                                # not enough time for vertical
                                if startRow < endRow:
                                    c = np.arange(startRow+1, startRow+remainingTime+1)
                                else:
                                    c = np.arange(startRow-remainingTime, startRow)[::-1]
                            t2 = np.arange(t1[-1]+1, t1[-1] + len(c)+1)
                            self.groundTruths[index,t2, c, endCol] += 1
            logging.info('End {0} iteration, cost {1}\n'.format(index, time.time() - startTimeIter))
        logging.info('finish generate, cost {0}'.format(time.time() - startTimeTotal))


    # maybe startPointsNum != endPointsNum
    def choosePoints(self, pointsNum):
        return np.random.choice(self.row*self.column, pointsNum, replace=False)


    # Nomalize groundTruths, for each element, if it is less than median, assign to 0; otherwise assign to 1.
    def statusNormalize(self):
        medianVal = np.median(self.groundTruths[self.groundTruths!=0])
        self.groundTruths[self.groundTruths>medianVal] = 1
        self.groundTruths[self.groundTruths<=medianVal] = 0


    # only save time after 30 seconds
    def dataProcess(self):
        self.statusNormalize()
        self.trainingSets = self.trainingSets[:,30:,:,:,:]
        self.groundTruths = self.groundTruths[:,30:,:,:]

    def chooseTimeClip(self):
        self.groundTruths = self.groundTruths[:, np.arange(0, self.groundTruths.shape[1], step=5), :,:,:]

    def generateLaunchingPoints(self):
        launchingPoints1 = np.array([
            [15, 0, 0],
            [15, 2, 0],
            [16, 0, 0],
            [16, 2, 0],  
        ])

        launchingPoints2 = np.array([
            [7, 10, 0],
            [7, 11, 0],
            [8, 10, 0],
            [8, 11, 0],
        ])

        launchingPoint3 = np.array([
            [23, 29, 0],
            [23, 30, 0],
            [24, 29, 0],
            [24, 30, 0],
        ])

        destination = np.array([
            [30, 9],
            [30, 10],
            [31, 9],
            [31, 10],
            [7, 22],
            [7, 25],
            [9, 22],
            [9, 25],
            [30, 9],
            [30, 10],
            [31, 9],
            [31, 10],
        ])

        launchingArea = np.array([launchingPoints1, launchingPoints2, launchingPoint3], dtype=np.float32)
        for i in range(len(launchingArea)):
            p = np.random.uniform(0.2, 0.8)
            launchingArea[i,:,2] = p
        launchingArea = np.concatenate(launchingArea, axis=0)
        
        return launchingArea, destination
        


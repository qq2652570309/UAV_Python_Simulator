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
        startPoints = self.choosePoints(self.startPointsNum)
        startPositions = list(map(lambda x: (x//self.column, x%self.column), startPoints))
        endPoints = self.choosePoints(self.endPointsNum)
        endPositions = list(map(lambda x: (x//self.column, x%self.column), endPoints))
        for index in range(self.iteration):
            startTimeIter = time.time()
            logging.info('At {0} iteration'.format(index))
            # startPoints = self.choosePoints(self.startPointsNum)
            # startPositions = list(map(lambda x: (x//self.column, x%self.column), startPoints))
            # endPoints = self.choosePoints(self.endPointsNum)
            # endPositions = list(map(lambda x: (x//self.column, x%self.column), endPoints))

            for startRow, startCol in startPositions:
                logging.info('   At start Point ({0}, {1})'.format(startRow, startCol))
                # set traning sets
                self.trainingSets[index,:-1,startRow,startCol,1] = np.random.uniform(0, 1)
                # generate ground truth
                for currentTime in range(self.time):
                    if currentTime == self.time -1:
                        continue

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



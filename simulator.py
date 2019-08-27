'''
# In channel, 0th is status that uav is launching at this second
# 1st is launching rate of this point
# 2nd and 3rd is (x, y) postion of destination point
'''
import time
import random
import logging
import numpy as np
from Area import Area

class Simulator:
    def __init__(self, iteration = 1, time=60, row=0, column=0,  uavNum=None, requiredDist=15, timeInterval=5):
        self.iteration = iteration
        self.row = row
        self.column = column
        self.time = time
        self.requiredDist = requiredDist
        self.timeInterval = timeInterval
        self.area = None
        self.uavNum = uavNum
        # In channel, 0th is status that uav is launching at this second
        # 1st is launching rate of this point
        # 2nd and 3rd is (x, y) postion of destination point
        self.trainingSets = np.zeros(shape=(self.iteration, self.time, self.row, self.column, 4), dtype=np.float32)
        self.groundTruths = np.zeros(shape=(self.iteration, self.time, self.row, self.column), dtype=np.float32)
        # record all launching and landing postions
        self.positions = np.zeros(shape=(self.iteration, 32, 32), dtype=np.float32)
        logging.info('finish init\n')


    def generate(self):
        startTimeTotal = time.time()
        
        for index in range(self.iteration):
            startTimeIter = time.time()
            logging.info('At {0} iteration'.format(index))

            self.area = Area()
            self.setColor(index, self.area.la, self.area.da)
            
            # startPositions, endPositions = self.generatePositions() 
            if self.uavNum == None:
                # startPositions = self.area.getLaunchPoint(low=0.1, high=0.8)
                startPositions = self.area.getLaunchPoint(low=1, high=1)
            else:
                # startPositions = self.area.getLaunchPoint(low=0.1, high=0.8, n=self.uavNum)
                startPositions = self.area.getLaunchPoint(low=1, high=1, n=self.uavNum)

            for startRow, startCol, launchingRate in startPositions:
                logging.info('   At start Point ({0}, {1})'.format(startRow, startCol))
                # set traning sets
                startRow = int(startRow)
                startCol = int(startCol)
                self.trainingSets[index,:,startRow,startCol,1] = launchingRate

                # generate ground truth
                for currentTime in range(self.time):
                    succ = np.random.uniform(0,1) <= self.trainingSets[index,currentTime,startRow,startCol,1]
                    if succ:
                        endRow, endCol = self.area.getDestination()
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

    def computeDistance(self, endPosition, startPositions, requiredDist):
        for startPoinst in startPositions:
            distance = np.abs(startPoinst[0]-endPosition[0]) + np.abs(startPoinst[1]-endPosition[1])
            if distance <= requiredDist :
                return False
        return True

    def setColor(self, index, startPositions, endPositions):
        for sp, ep in zip(startPositions, endPositions):
            self.positions[index, sp[0], sp[1]] = 0.2
            self.positions[index, ep[0], ep[1]] = 0.5


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.disabled = True

    logging.basicConfig(filename='log.txt', format='%(levelname)s:%(message)s', level=logging.INFO)

    logging.info('Started')
    startTimeIter = time.time()
    # s = Simulator(iteration=2, row=4, column=4, time=5, startPointsNum=3, endPointsNum=3)
    # s = Simulator(iteration=10000, row=16, column=16, time=60, startPointsNum=15, endPointsNum=15)
    s = Simulator(iteration=100, row=32, column=32, time=120, uavNum=2)
    s.generate()

    logging.info('Finished')
    np.save('data/test_trainingSets.npy', s.trainingSets)
    np.save('data/test_groundTruths.npy', s.groundTruths)
    np.save('data/positions.npy', s.positions)
    print('total time: ', time.time() - startTimeIter)
    # logging.info('trainingSets: \n{0}'.format(s.trainingSets))
    # logging.info('groundTruths: \n{0}'.format(s.groundTruths))
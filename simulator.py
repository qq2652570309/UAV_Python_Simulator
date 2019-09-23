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
    def __init__(self, batch = 1, time=60, row=0, column=0,  uavNum=None, requiredDist=15, timeInterval=5):
        self.batch = batch
        self.row = row
        self.column = column
        self.time = time
        self.requiredDist = requiredDist
        self.timeInterval = timeInterval
        self.area = Area(0,1)
        self.uavNum = uavNum
        # In channel, 0th is status that uav is launching at this second
        # 1st is launching rate of this point
        # 2nd and 3rd is (x, y) postion of destination point
        self.trainingSets = np.zeros(shape=(self.batch, self.time, self.row, self.column, 2), dtype=np.float32)
        self.groundTruths = np.zeros(shape=(self.batch, self.time, self.row, self.column), dtype=np.float32)
        # record all launching and landing postions
        self.totalFlyingTime = 0
        self.totalUavNum = 0
        # logging.info('finish init\n')


    def generate(self):
        startTimeTotal = time.time()
        
        for batch_idx in range(self.batch):
            startTimeIter = time.time()
            logging.info('At {0} iteration'.format(batch_idx))

            self.area.refresh(mapSize=self.row, areaSize=3, num=10)
            
            for currentTime in range(self.time):
                # generate ground truth
                startPositions = self.area.getLaunchPoint()
                for startRow, startCol, launchingRate in startPositions:
                    startRow = int(startRow)
                    startCol = int(startCol)
                    
                    self.trainingSets[batch_idx,:,startRow,startCol,0] = launchingRate
                    succ = np.random.uniform(0,1) <= self.trainingSets[batch_idx,currentTime,startRow,startCol,0]
                    if succ:
                        self.totalUavNum += 1
                        endRow, endCol = self.area.getDestination()
                        remainingTime = self.time - currentTime
                        
                        # logging.info('      At time {0}, ({1}, {2}) --> ({3}, {4})'.format(currentTime, startRow, startCol, endRow, endCol))
                        flyingTime = 0
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
                        self.groundTruths[batch_idx,t1,startRow,r] += 1
                        remainingTime -= len(r)
                        self.totalFlyingTime += len(r)

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
                            self.groundTruths[batch_idx,t2, c, endCol] += 1
                            self.totalFlyingTime += len(c)
            logging.info('End {0} iteration, cost {1}\n'.format(batch_idx, time.time() - startTimeIter))
        # logging.info('finish generate, cost {0}'.format(time.time() - startTimeTotal))'''



if __name__ == "__main__":
    logger = logging.getLogger()
    logger.disabled = True

    logging.basicConfig(filename='log.txt', format='%(levelname)s:%(message)s', level=logging.INFO)

    logging.info('Started')
    startTimeIter = time.time()
    # s = Simulator(iteration=2, row=4, column=4, time=5, startPointsNum=3, endPointsNum=3)
    # s = Simulator(iteration=10000, row=16, column=16, time=60, startPointsNum=15, endPointsNum=15)
    s = Simulator(batch=10, row=100, column=100, time=240, timeInterval=5)
    s.generate()
    print('avg flying time: ', s.totalFlyingTime/s.totalUavNum)

    logging.info('Finished')
    print('total time: ', time.time() - startTimeIter)
    print('training shape: ', s.trainingSets.shape)
    print('groundTruth shape: ', s.groundTruths.shape)
    np.save('data/trainingSets.npy',s.trainingSets)
    np.save('data/groundTruths.npy',s.groundTruths)
    # logging.info('trainingSets: \n{0}'.format(s.trainingSets))
    # logging.info('groundTruths: \n{0}'.format(s.groundTruths))
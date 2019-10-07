'''
# In channel
# 0th and 1st is (x, y) location launching point
# 2nd and 3rd is (x, y) location of landing point
'''
import time
import os
import random
import logging
import numpy as np
from Area import Area


class Simulator4:
    def __init__(self, batch = 1, time=200, mapSize=100,  taskNum=15):
        self.batch = batch
        self.map_size = mapSize
        self.time = time
        self.task_num = taskNum
        self.area = Area(0,1)
        # In channel, 1st and 2nd are (x, y) launching location, 
        # 3rd and 4th are (x, y) destination location
        # 5th is time
        self.tasks = np.zeros(shape=(batch, 100, taskNum, 5), dtype=int)
        self.trajectors = np.zeros(shape=(batch, 110, mapSize, mapSize), dtype=int)
        self.Rfeature = np.zeros(shape=(self.batch, mapSize, mapSize, 2), dtype=np.float32)
        self.totalFlyingTime = 0
        self.totalUavNum = 0
        if os.path.exists('log.txt'):
            os.remove('log.txt')
        logging.basicConfig(filename='log.txt', format='%(levelname)s:%(message)s', level=logging.INFO)


    def generate(self):
        for batch_idx in range(self.batch):
            startTimeIter = time.time()
            trajectors = np.zeros(shape=(self.time, self.map_size, self.map_size), dtype=int)

            self.area.refresh(mapSize=self.map_size, areaSize=3, num=10)
            start_time = random.choice(range(0, 80))
            self.drawPatten(batch_idx)

            # time iteration
            for currentTime in range(self.time):
                if currentTime >= start_time + 110:
                    break;
                
                # task iteration
                startPositions = self.area.getLaunchPoint(n=self.task_num)
                for task_idx, task_val in zip(range(len(startPositions)), startPositions):
                    startRow, startCol, launchingRate = task_val
                    startRow = int(startRow)
                    startCol = int(startCol)
                    succ = np.random.uniform(0,1) <= launchingRate
                    
                    # if there is a launching UAV
                    if currentTime >= start_time + 10 and currentTime < start_time + 110:
                        time_idx = currentTime - (start_time + 10)
                        self.tasks[batch_idx,time_idx,task_idx,4] = currentTime
                    if succ:
                        self.totalUavNum += 1
                        endRow, endCol = self.area.getDestination()

                        # add info into channel
                        if currentTime >= start_time + 10 and currentTime < start_time + 110:
                            self.tasks[batch_idx,time_idx,task_idx,0] = startRow
                            self.tasks[batch_idx,time_idx,task_idx,1] = startCol
                            self.tasks[batch_idx,time_idx,task_idx,2] = endRow
                            self.tasks[batch_idx,time_idx,task_idx,3] = endCol
                        
                        # add launching rate to map
                        self.Rfeature[batch_idx, startRow, startCol, 0] = launchingRate
                        # add landing rate to map
                        self.Rfeature[batch_idx, endRow, endCol, 0] = 0.3

                        remainingTime = self.time - currentTime
                        
                        
                        midCol = int(round((startCol + endCol) / 2))
                        midRow1 = int(startRow)
                        midRow2 = int(endRow)
                        
                        
                        # first horizontal movement
                        if remainingTime >= abs(startCol-midCol)+1 :
                            # enough time for horizontal
                            if startCol < midCol :
                                r1 =  np.arange(startCol, midCol+1, dtype=int)
                            else:
                                r1 = np.arange(midCol, startCol+1, dtype=int)[::-1]
                        else:
                            # not enough time for horizontal
                            if startCol < midCol:
                                r1 = np.arange(startCol, startCol+remainingTime, dtype=int)
                            else:
                                r1 = np.arange(startCol-remainingTime+1, startCol+1, dtype=int)[::-1]
                        t1 = np.arange(currentTime, currentTime+len(r1))
                        trajectors[t1,startRow,r1] += 1
                        remainingTime -= len(r1)
                        self.totalFlyingTime += len(r1)
                        
                        # vertical movement 
                        if remainingTime > 0 :
                            # exists time for vertical
                            if remainingTime >= abs(midRow1-midRow2) :
                                # enough time for vertical
                                if midRow1 < midRow2:
                                    c = np.arange(midRow1+1, midRow2+1, dtype=int)
                                else:
                                    c = np.arange(midRow2, midRow1, dtype=int)[::-1]
                            else:
                                # not enough time for vertical
                                if midRow1 < midRow2:
                                    c = np.arange(midRow1+1, midRow1+remainingTime+1, dtype=int)
                                else:
                                    c = np.arange(midRow1-remainingTime, midRow1, dtype=int)[::-1]
                            t2 = np.arange(t1[-1]+1, t1[-1] + len(c)+1, dtype=int)
                            trajectors[t2, c, midCol] += 1
                            remainingTime -= len(c)
                            self.totalFlyingTime += len(c)
                        
                        # second horizontal movement
                        if remainingTime > 0 :
                            # exists time for horizontal
                            if remainingTime >= abs(midCol-endCol):
                                # enough time for horizontal
                                if midCol < endCol :
                                    r2 =  np.arange(midCol+1, endCol, dtype=int)
                                else:
                                    r2 = np.arange(endCol, midCol, dtype=int)[::-1]
                            else:
                                # not enough time for horizontal
                                if midCol < endCol:
                                    r2 = np.arange(midCol+1, midCol+remainingTime, dtype=int)
                                else:
                                    r2 = np.arange(midCol-remainingTime+1, midCol, dtype=int)[::-1]
                            # check if uav has vertical movement
                            if len(t2) > 0:
                                t3 = np.arange(t2[-1], t2[-1] + len(r2), dtype=int)
                            else:
                                t3 = np.arange(t1[-1]+1, t1[-1] + len(r2)+1, dtype=int)
                            trajectors[t3,endRow,r2] += 1
                            remainingTime -= len(r2)
                            self.totalFlyingTime += len(r2)
            logging.info('End {0} iteration, cost {1}'.format(batch_idx, time.time() - startTimeIter))
            print('End {0} iteration, cost {1}\n'.format(batch_idx, time.time() - startTimeIter))
            logging.info('{0} batch, start time {1}\n'.format(batch_idx, start_time))
            self.trajectors[batch_idx] = trajectors[start_time:start_time+110]


    def drawPatten(self, batch_idx):
        startPositions = self.area.getLaunchPoint()
        for startRow, startCol, _ in startPositions:
            for endRow, endCol in self.area.getDestination(allPoints=True):
                startRow, startCol = int(startRow), int(startCol)
                endRow, endCol = int(endRow), int(endCol)
                if startCol < endCol :
                    r =  np.arange(startCol, endCol+1)
                else:
                    r = np.arange(endCol, startCol+1)[::-1]
                self.Rfeature[batch_idx, startRow, r, 1] = 1
                if startRow < endRow:
                    c = np.arange(startRow+1, endRow+1)
                else:
                    c = np.arange(endRow, startRow)[::-1]
                self.Rfeature[batch_idx, c, endCol, 1] = 1
                
                
                midCol = round((startCol + endCol) / 2)
                midRow1 = startRow
                midRow2 = endRow
                # first horizontal movement
                if startCol < midCol :
                    r1 =  np.arange(startCol, midCol+1, dtype=int)
                else:
                    r1 = np.arange(midCol, startCol+1, dtype=int)[::-1]
                self.Rfeature[batch_idx, startRow, r1, 1] = 1
                
                # vertical movement 
                # enough time for vertical
                if midRow1 < midRow2:
                    c = np.arange(midRow1+1, midRow2+1, dtype=int)
                else:
                    c = np.arange(midRow2, midRow1, dtype=int)[::-1]
                self.Rfeature[batch_idx, c, midCol, 1] = 1
                
                # second horizontal movement
                if midCol < endCol :
                    r2 =  np.arange(midCol+1, endCol, dtype=int)
                else:
                    r2 = np.arange(endCol, midCol, dtype=int)[::-1]
                self.Rfeature[batch_idx, endRow, r2, 1] = 1



if __name__ == "__main__":
    logger = logging.getLogger()
    logger.disabled = True

    logging.basicConfig(filename='log.txt', format='%(levelname)s:%(message)s', level=logging.INFO)

    logging.info('Started')
    startTimeIter = time.time()
    
    s = Simulator4(batch=1, mapSize=100, time=200)
    # s.generate()
    # print('UAV Avg Flying Time: ', s.totalFlyingTime/s.totalUavNum)

    logging.info('Finished')
    np.save('data/tasks.npy', s.tasks)
    np.save('data/trajectors.npy', s.trajectors)
    # print('Simulation Total Time: ', time.time() - startTimeIter)

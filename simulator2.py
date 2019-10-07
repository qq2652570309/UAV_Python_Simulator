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


class Simulator2:
    def __init__(self, batch = 1, time=200, mapSize=100, taskNum=15, trajectoryTime=110, taskTime=50):
        self.batch = batch
        self.map_size = mapSize
        self.time = time
        self.task_num = taskNum
        self.trajectoryTime = trajectoryTime
        self.taskTime = taskTime
        self.area = Area(0,1)
        # In channel, 1st and 2nd are (x, y) launching location, 
        # 3rd and 4th are (x, y) destination location
        # 5th is time
        self.tasks = np.zeros(shape=(batch, taskTime, taskNum, 5), dtype=int)
        self.trajectors = np.zeros(shape=(batch, trajectoryTime, mapSize, mapSize), dtype=int)
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
                if currentTime >= start_time + self.trajectoryTime:
                    break
                
                if currentTime == start_time + 10 + 25:
                    self.area.updateLaunchingRate()
                
                # task iteration
                startPositions = self.area.getLaunchPoint(n=self.task_num)
                for task_idx, task_val in zip(range(len(startPositions)), startPositions):
                    startRow, startCol, launchingRate = task_val
                    startRow = int(startRow)
                    startCol = int(startCol)
                    succ = np.random.uniform(0,1) <= launchingRate
                    
                    # if there is a launching UAV
                    if currentTime >= start_time + 10 and currentTime < start_time + self.trajectoryTime:
                        time_idx = currentTime - (start_time + 10)
                        self.tasks[batch_idx,time_idx,task_idx,4] = currentTime
                    if succ:
                        self.totalUavNum += 1
                        endRow, endCol = self.area.getDestination()

                        # add info into channel
                        if currentTime >= start_time + 10 and currentTime < start_time + self.trajectoryTime:
                            self.tasks[batch_idx,time_idx,task_idx,0] = startRow
                            self.tasks[batch_idx,time_idx,task_idx,1] = startCol
                            self.tasks[batch_idx,time_idx,task_idx,2] = endRow
                            self.tasks[batch_idx,time_idx,task_idx,3] = endCol
                        
                        # add launching rate to map
                        self.Rfeature[batch_idx, startRow, startCol, 0] = launchingRate
                        # add landing rate to map
                        self.Rfeature[batch_idx, endRow, endCol, 0] = 0.3

                        remainingTime = self.time - currentTime
                        
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
                        trajectors[t1,startRow,r] += 1
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
                            trajectors[t2, c, endCol] += 1
                            self.totalFlyingTime += len(c)
            logging.info('End {0} iteration, cost {1}'.format(batch_idx, time.time() - startTimeIter))
            print('End {0} iteration, cost {1}\n'.format(batch_idx, time.time() - startTimeIter))
            logging.info('{0} batch, start time {1}\n'.format(batch_idx, start_time))
            self.trajectors[batch_idx] = trajectors[start_time:start_time + self.trajectoryTime]


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



if __name__ == "__main__":
    logger = logging.getLogger()
    logger.disabled = True

    logging.basicConfig(filename='log.txt', format='%(levelname)s:%(message)s', level=logging.INFO)

    logging.info('Started')
    startTimeIter = time.time()
    
    s = Simulator2(batch=1, mapSize=100, time=200)
    # s.generate()
    # print('UAV Avg Flying Time: ', s.totalFlyingTime/s.totalUavNum)

    logging.info('Finished')
    np.save('data/tasks.npy', s.tasks)
    np.save('data/trajectors.npy', s.trajectors)
    # print('Simulation Total Time: ', time.time() - startTimeIter)

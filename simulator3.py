'''
# 
'''
import time
import os
import random
import logging
import numpy as np
from Area import Area


class Simulator3:
    def __init__(self, batch = 1, time=200, mapSize=100, taskNum=15, trajectoryTime=110, taskTime=100):
        self.batch = batch
        self.map_size = mapSize
        self.time = time
        self.task_num = taskNum
        self.trajectoryTime = trajectoryTime
        self.taskTime = taskTime
        self.area = Area(0,1)
        # In channel, 1st and 2nd are (x, y) launching location, 
        # 3rd and 4th are (x, y) destination location
        self.tasks = np.zeros(shape=(batch, taskTime, taskNum, 5), dtype=int)
        self.trajectors = np.zeros(shape=(batch, trajectoryTime, mapSize, mapSize), dtype=int)
        self.Rfeatrue = np.zeros(shape=(batch, mapSize, mapSize, 2), dtype=np.float32)
        self.totalFlyingTime = 0
        self.totalUavNum = 0
        if os.path.exists('./log.txt'):
            os.remove('log.txt')
        
    
    def generate(self):
        for batch_idx in range(self.batch):
            startTimeIter = time.time()
            trajectors = np.zeros(shape=(self.time, self.map_size, self.map_size), dtype=int)

            self.area.refresh(mapSize=self.map_size, areaSize=3, num=10)
            self.drawPatten(batch_idx)
            start_time = random.choice(range(0, 80))
            
            # time iteration
            for currentTime in range(self.time):
                if (currentTime >= start_time + self.trajectoryTime):
                    break
                # task iteration
                startPositions = self.area.getLaunchPoint(n=self.task_num)
                for task_idx, task_val in zip(range(len(startPositions)), startPositions):
                    startRow, startCol, launchingRate = task_val
                    if currentTime >= start_time + 10 and currentTime < start_time + 10 + self.taskTime:
                        time_idx = currentTime - (start_time + 10)
                        self.tasks[batch_idx,time_idx,task_idx,4] = currentTime
                    startRow = int(startRow)
                    startCol = int(startCol)
                    succ = np.random.uniform(0,1) <= launchingRate
                    
                    # if there is a launching UAV
                    if succ:
                        self.totalUavNum += 1
                        endRow, endCol = self.area.getDestination()
                        self.Rfeatrue[batch_idx, startRow, startCol, 0] = launchingRate
                        self.Rfeatrue[batch_idx, endRow, endCol, 0] = 0.3

                        # add info into channel
                        if currentTime >= start_time + 10 and currentTime < start_time + 10 + self.taskTime:
                            # print(time_idx, task_idx, currentTime, self.start_time + 10, self.start_time + 70)
                            self.tasks[batch_idx,time_idx,task_idx,0] = startRow
                            self.tasks[batch_idx,time_idx,task_idx,1] = startCol
                            self.tasks[batch_idx,time_idx,task_idx,2] = endRow
                            self.tasks[batch_idx,time_idx,task_idx,3] = endCol
                            # self.tasks[batch_idx,time_idx,task_idx,4] = currentTime

                        remainingTime = self.time - currentTime
                        
                        # routing part
                        if remainingTime >= abs(startRow-endRow)+1 :
                            # enough time for vertical
                            if startRow < endRow:
                                c = np.arange(startRow, endRow+1)
                            else:
                                c = np.arange(endRow, startRow+1)[::-1]
                        else:
                            # not enough time for vertical
                            if startRow < endRow:
                                c = np.arange(startRow, startRow+remainingTime)
                            else:
                                c = np.arange(startRow-remainingTime+1, startRow+1)[::-1]


                        t1 = np.arange(currentTime, currentTime+len(c))
                        trajectors[t1,c,startCol] += 1
                        remainingTime -= len(c)
                        self.totalFlyingTime += len(c)

                        if remainingTime > 0 :
                            if remainingTime >= abs(startCol-endCol) :
                                # enough time for horizontal
                                if startCol < endCol :
                                    r =  np.arange(startCol+1, endCol+1)
                                else:
                                    r = np.arange(endCol, startCol)[::-1]
                            else:
                                # not enough time for horizontal
                                if startCol < endCol:
                                    r = np.arange(startCol+1, startCol+remainingTime+1)
                                else:
                                    r = np.arange(startCol-remainingTime, startCol)[::-1]
                            t2 = np.arange(t1[-1]+1, t1[-1] + len(r)+1)
                            trajectors[t2,endRow,r] += 1
                            remainingTime -= len(r)
                            self.totalFlyingTime += len(r)

            logging.info('End {0} iteration, cost {1}'.format(batch_idx, time.time() - startTimeIter))
            print('End {0} iteration, cost {1}\n'.format(batch_idx, time.time() - startTimeIter))
            logging.info('{0} batch, start time {1}\n'.format(batch_idx, start_time))
            self.trajectors[batch_idx] = trajectors[start_time:start_time+self.trajectoryTime]

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
                self.Rfeatrue[batch_idx, startRow, r, 1] = 1
                if startRow < endRow:
                    c = np.arange(startRow+1, endRow+1)
                else:
                    c = np.arange(endRow, startRow)[::-1]
                self.Rfeatrue[batch_idx, c, endCol, 1] = 1

if __name__ == "__main__":
    s = Simulator1(batch=1, mapSize=100)
    s.generate()
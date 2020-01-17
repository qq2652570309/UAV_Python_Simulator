'''
# main net + subnet
'''
import time
import os
import random
import logging
import numpy as np
from Area import Area

random.seed(0)
np.random.seed(0)

class SimulatorMainNet:
    def __init__(self, batch = 1, time=200, mapSize=100, taskNum=15, trajectoryTime=70, taskTime=60):
        self.batch = batch
        self.map_size = mapSize
        self.time = time
        self.task_num = taskNum
        self.area = Area(0,1)
        self.trajectoryTime = trajectoryTime
        self.taskTime = taskTime
        '''------------main net-----------------'''
        # channels: mainTaskList[0] and mainTaskList[1] is launching location
        # channels: mainTaskList[2] and mainTaskList[3] is landing location 
        # channels: mainTaskList[5] is launching time
        # mainTaskList = (3000, 60, 15, 5)
        self.mainTaskList = np.zeros(shape=(batch, taskTime, taskNum, 5), dtype=int)
        # every timestep, number of uav on each grid
        # used for generate density map (label) and init density (input)
        self.trajectors = np.zeros(shape=(batch, trajectoryTime, mapSize, mapSize), dtype=int)
        # subOutput = (3000, 60, 100, 100), tasklist as input for MainNet
        self.subOutput = np.zeros(shape=(batch, taskTime, mapSize, mapSize), dtype=float)
        # Rnet input
        self.Rfeature = np.zeros(shape=(batch, mapSize, mapSize, 2), dtype=np.float32)
        '''------------sub net-----------------'''
        # subTaskList = (3000*60, 15, 5), tasklist as input for SubNet
        self.subTaskList = np.zeros(shape=(batch * taskTime, taskNum, 5), dtype=float)
        # subLabel = (3000*60, 100, 100), as label for SubNet
        self.subLabel = np.zeros(shape=(batch * taskTime, mapSize, mapSize), dtype=float)
        self.counter = np.zeros(shape=(batch * taskTime, mapSize, mapSize), dtype=int)
        self.startValue = 0.25
        self.endValue = 0.75
        '''------------statistic-----------------'''
        self.totalFlyingTime = 0
        self.totalUavNum = 0
        if os.path.exists('./log.txt'):
            os.remove('log.txt')
    
    def generate(self):
        
        for batch_idx in range(self.batch):
            startTimeIter = time.time()
            trajectors = np.zeros(shape=(self.time, self.map_size, self.map_size), dtype=int)

            self.area.refresh(mapSize=self.map_size, areaSize=3, num=10)
            self.drawPatten_horizontal_vertical(batch_idx)
            start_time = random.choice(range(70, 80))
            
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
                        self.mainTaskList[batch_idx,time_idx,task_idx,4] = currentTime
                        self.subTaskList[batch_idx*60+time_idx, task_idx, 4] = currentTime
                    startRow = int(startRow)
                    startCol = int(startCol)
                    succ = np.random.uniform(0,1) <= launchingRate
                    
                    # if there is a launching UAV
                    if succ:
                        self.totalUavNum += 1
                        endRow, endCol = self.area.getDestination()
                        self.Rfeature[batch_idx, startRow, startCol, 0] = launchingRate
                        self.Rfeature[batch_idx, endRow, endCol, 0] = 0.3

                        # add info into channel
                        if currentTime >= start_time + 10 and currentTime < start_time + 10 + self.taskTime:

                            self.mainTaskList[batch_idx,time_idx,task_idx,0] = startRow
                            self.mainTaskList[batch_idx,time_idx,task_idx,1] = startCol
                            self.mainTaskList[batch_idx,time_idx,task_idx,2] = endRow
                            self.mainTaskList[batch_idx,time_idx,task_idx,3] = endCol

                            self.sliceTaskMap(batch_idx, time_idx, task_idx, startRow, startCol, endRow, endCol)
                        trajectors = self.horizontal_vertical(startRow=startRow, startCol=startCol, 
                                                            endRow=endRow, endCol=endCol, 
                                                            currentTime=currentTime, trajectors=trajectors)
            self.trajectors[batch_idx] = trajectors[start_time:start_time+self.trajectoryTime]

            logging.info('End {0} iteration, cost {1}'.format(batch_idx, time.time() - startTimeIter))
            print('End {0} iteration, cost {1}\n'.format(batch_idx, time.time() - startTimeIter))
            logging.info('{0} batch, start time {1}\n'.format(batch_idx, start_time))

        self.subLabel = np.nan_to_num(self.subLabel / self.counter)
        for b in range(self.batch):
            for t in range(self.taskTime):
                self.subOutput[b, t] = self.subLabel[b*self.taskTime+t]
            

    def horizontal_vertical(self, startRow, startCol, endRow, endCol, currentTime, trajectors):
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
        return trajectors

    def vertical_horizontal(self, startRow, startCol, endRow, endCol, currentTime, trajectors):
        remainingTime = self.time - currentTime
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
        return trajectors

    def drawPatten_horizontal_vertical(self, batch_idx):
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

    def drawPatten_vertical_horizontal(self, batch_idx):
        startPositions = self.area.getLaunchPoint()
        for startRow, startCol, _ in startPositions:
            for endRow, endCol in self.area.getDestination(allPoints=True):
                startRow, startCol = int(startRow), int(startCol)
                endRow, endCol = int(endRow), int(endCol)
                if startRow < endRow:
                    c = np.arange(startRow, endRow+1)
                else:
                    c = np.arange(endRow, startRow+1)[::-1]
                self.Rfeature[batch_idx, c, startCol] = 1
                if startCol < endCol :
                    r =  np.arange(startCol+1, endCol+1)
                else:
                    r = np.arange(endCol, startCol)[::-1]
                self.Rfeature[batch_idx, endRow, r] = 1

    def sliceTaskMap(self, batch_idx, time_idx, task_idx, startRow, startCol, endRow, endCol):
        i = batch_idx*60 + time_idx

        self.subTaskList[i, task_idx, 0] = startRow 
        self.subTaskList[i, task_idx, 1] = startCol
        self.subTaskList[i, task_idx, 2] = endRow
        self.subTaskList[i, task_idx, 3] = endCol

        # compute each step value
        pathLen = abs(startRow-endRow) + abs(endCol-startCol) + 1
        step = (self.endValue-self.startValue)/(pathLen-1)
        steps = np.around(np.arange(start=self.startValue, stop=self.endValue+step, step=step), 2)

        if startCol < endCol :
            r =  np.arange(startCol, endCol+1)
        else:
            r = np.arange(endCol, startCol+1)[::-1]
        # self.subLabel[i, task_idx, startRow, r] += 1
        self.subLabel[i, startRow, r] += steps[np.arange(0, len(r))]
        self.counter[i, startRow, r] += 1
        stepIndex = len(r)

        if startRow < endRow:
            c = np.arange(startRow+1, endRow+1)
        else:
            c = np.arange(endRow, startRow)[::-1]
        # self.subLabel[i, task_idx, c, endCol] += 1
        self.subLabel[i, c, endCol] += steps[np.arange(stepIndex, stepIndex+len(c))]
        self.counter[i, c, endCol] += 1



if __name__ == "__main__":
    s = SimulatorMainNet(batch=30, mapSize=100)
    s.generate()
    print("\n--------SubNet--------")
    print('subTaskList: {0}'.format(s.subTaskList.shape))
    print('subLabel:    {0}'.format(s.subLabel.shape))
    print('counter:     {0}'.format(s.counter.shape))
    print("--------MainNet--------")
    print('mainTaskList: {0}'.format(s.mainTaskList.shape))
    print('trajectors:   {0}'.format(s.trajectors.shape))
    print('subOutput :   {0}'.format(s.subOutput.shape))
    print('Rfeature:     {0}'.format(s.Rfeature.shape))


    # np.save('../../../data/zzhao/uav_regression/{0}/{1}.npy'.format('test', 'mainTaskList'), s.mainTaskList)
    # np.save('../../../data/zzhao/uav_regression/{0}/{1}.npy'.format('test', 'trajectors'), s.trajectors)
    # np.save('../../../data/zzhao/uav_regression/{0}/{1}.npy'.format('test', 'Rfeature'), s.Rfeature)

    # np.save('../../../data/zzhao/uav_regression/{0}/{1}.npy'.format('test', 'subTaskList'), s.subTaskList)
    # np.save('../../../data/zzhao/uav_regression/{0}/{1}.npy'.format('test', 'subLabel'), s.subLabel)
    # np.save('../../../data/zzhao/uav_regression/{0}/{1}.npy'.format('test', 'counter'), s.counter)
    

    



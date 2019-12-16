'''
# main net + subnet
'''


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import time
import os
import random
import logging
import numpy as np
from Area import Area

random.seed(0)
np.random.seed(0)

class SimulatorTest:
    def __init__(self, batch = 1, time=350, mapSize=100, taskNum=15, trajectoryTime=70, taskTime=60, restrictStart=-1, restrctEnd=-1):
        self.batch = batch
        self.map_size = mapSize
        self.time = time
        self.task_num = taskNum
        self.area = Area(mapSize=100, areaSize=3, areaNum=10)
        self.trajectoryTime = trajectoryTime
        self.taskTime = taskTime
        self.restrictStart = restrictStart
        self.restrctEnd = restrctEnd
        self.start_time = 0
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
        # no fly zone
        self.NFZ = np.zeros(shape=(batch, mapSize, mapSize))
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
        
        self.testArea = np.zeros(shape=(batch, mapSize, mapSize))
        
    
    def generate(self):
        
        for batch_idx in range(self.batch):
            startTimeIter = time.time()
            trajectors = np.zeros(shape=(self.time, self.map_size, self.map_size), dtype=int)
            # self.drawPatten_horizontal_vertical(batch_idx)
            # self.area.refresh(mapSize=self.map_size, areaSize=3, num=10)
            self.area.refresh(batch=batch_idx)
            start_time = random.choice(range(70, 80))
            self.start_time = start_time

            noFlyZone = self.area.getNoFlyZone()
            x1, y1 = noFlyZone[0]
            x3, y3 = noFlyZone[2]
            self.NFZ[batch_idx, x1:x3, y1:y3] = 120
            
            # time iteration
            for currentTime in range(self.time):
                if (currentTime >= start_time + self.trajectoryTime):
                    break
                # task iteration
                startPositions = self.area.getLaunchPoint(n=self.task_num)

                for task_idx, task_val in zip(range(len(startPositions)), startPositions):
                    startRow, startCol, launchingRate = task_val
                    time_idx = -1
                    if currentTime >= start_time + 10 and currentTime < start_time + 10 + self.taskTime:
                        time_idx = currentTime - (start_time + 10)
                        self.mainTaskList[batch_idx,time_idx,task_idx,4] = currentTime
                        self.subTaskList[batch_idx*60+time_idx, task_idx, 4] = currentTime
                    startRow = int(startRow)
                    startCol = int(startCol)
                    succ = np.random.uniform(0,1) <= launchingRate
                    if self.restrictStart != -1 :
                        succ = succ and (currentTime < (start_time + 10) or (start_time + self.restrictStart + 10) <= currentTime)
                    if self.restrctEnd != -1 :
                        succ = succ and currentTime < (start_time + self.restrctEnd+10)

                    # if there is a launching UAV
                    if succ:
                        self.totalUavNum += 1
                        endRow, endCol = self.area.getDestination()
                        self.Rfeature[batch_idx, startRow, startCol, 0] = launchingRate
                        self.Rfeature[batch_idx, endRow, endCol, 0] = 0.3
                        # whether current time is in task time interval
                        isInterval  = True if currentTime >= start_time + 10 and currentTime < start_time + 10 + self.taskTime else False
                        path = []
                        pathLen = []

                        if isInterval:
                            self.mainTaskList[batch_idx,time_idx,task_idx,0] = startRow
                            self.mainTaskList[batch_idx,time_idx,task_idx,1] = startCol
                            self.mainTaskList[batch_idx,time_idx,task_idx,2] = endRow
                            self.mainTaskList[batch_idx,time_idx,task_idx,3] = endCol

                        # [ and ]
                        if noFlyZone[0, 1] <= startCol <= noFlyZone[2, 1] and noFlyZone[0, 1] <= endCol <= noFlyZone[2, 1]:
                            path, pathLen = self.verticalRouting(startRow, startCol, endRow, endCol, noFlyZone)
                            trajectors = self.threeStageRouting(path, pathLen, currentTime, batch_idx, time_idx, isInterval, trajectors)
                        # |冖| and |_|
                        elif noFlyZone[0, 0] <= startRow <= noFlyZone[2, 0] and noFlyZone[0, 0] <= endRow <= noFlyZone[2, 0] :
                            path, pathLen = self.horizontalRouting(startRow, startCol, endRow, endCol, noFlyZone)
                            trajectors = self.threeStageRouting(path, pathLen, currentTime, batch_idx, time_idx, isInterval, trajectors)
                        # modify routing
                        else:
                            def isHorizontalCross():
                                if not noFlyZone[0, 0] <= startRow <= noFlyZone[2, 0]:
                                    return False
                                uav_left = min(startCol, endCol)
                                uav_right = max(startCol, endCol)
                                nfz_left = noFlyZone[0,1]
                                nfz_right = noFlyZone[1,1]
                                if uav_left <= nfz_left <= uav_right <= nfz_right:
                                    return True
                                if nfz_left <= uav_left <= nfz_right <= uav_right:
                                    return True
                                if uav_left <= nfz_left < nfz_right <= uav_right:
                                    return True
                                return False
                            
                            def isVerticalCross():
                                if not noFlyZone[0,1] <= endCol <= noFlyZone[2,1]:
                                    return False
                                uav_up = min(startRow, endRow)
                                uav_down = max(startRow, endRow)
                                nfz_up = noFlyZone[0,0]
                                nfz_down = noFlyZone[2,0]
                                if uav_up <= nfz_up < nfz_down <= uav_down:
                                    return True
                                return False 
                        
                            if isHorizontalCross() or isVerticalCross():
                                # vertically move first, horizontally move second
                                trajectors = self.vertical_horizontal(startRow, startCol, endRow, endCol, currentTime, trajectors)
                                if isInterval:
                                    self.sliceTaskMap(batch_idx, time_idx, task_idx, startRow, startCol, endRow, endCol, horizontal=False)
                            else:
                                # horizontally move first, vertically move second
                                trajectors = self.horizontal_vertical(startRow, startCol, endRow, endCol, currentTime, trajectors)
                                if isInterval:
                                    self.sliceTaskMap(batch_idx, time_idx, task_idx, startRow, startCol, endRow, endCol, horizontal=True)

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

    # gnerate subnet label without no fly zone routing
    def sliceTaskMap(self, batch_idx, time_idx, task_idx, startRow, startCol, endRow, endCol, horizontal=False):
        i = batch_idx*60 + time_idx

        self.subTaskList[i, task_idx, 0] = startRow 
        self.subTaskList[i, task_idx, 1] = startCol
        self.subTaskList[i, task_idx, 2] = endRow
        self.subTaskList[i, task_idx, 3] = endCol

        # compute each step value
        pathLen = abs(startRow-endRow) + abs(endCol-startCol) + 1
        step = (self.endValue-self.startValue)/(pathLen-1)
        steps = np.around(np.arange(start=self.startValue, stop=self.endValue+step, step=step), 2)

        if horizontal:
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
        else:
            if startRow < endRow:
                c = np.arange(startRow+1, endRow+1)
            else:
                c = np.arange(endRow, startRow)[::-1]
            # self.subLabel[i, task_idx, c, endCol] += 1
            self.subLabel[i, c, endCol] += steps[np.arange(0, len(c))]
            self.counter[i, c, endCol] += 1
            stepIndex = len(c)

            if startCol < endCol :
                r =  np.arange(startCol, endCol+1)
            else:
                r = np.arange(endCol, startCol+1)[::-1]
            # self.subLabel[i, task_idx, startRow, r] += 1
            self.subLabel[i, startRow, r] += steps[np.arange(stepIndex, stepIndex+len(r))]
            self.counter[i, startRow, r] += 1

    
    # avoid no fly zone with routing |冖| or |_|
    def horizontalRouting(self, sr, sc, er, ec, noFlyZone):
        R1 = noFlyZone[0, 0]
        R2 = noFlyZone[2, 0]
        
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

    # avoid no fly zone with routing [ or ]
    def verticalRouting(self, sr, sc, er, ec, noFlyZone):
        C1 = noFlyZone[0, 1]
        C2 = noFlyZone[2, 1]
        
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

    # draw various paths after avoided no fly zone
    def threeStageRouting(self, path, pathLen, currentTime, batch_idx, time_idx, isInterval, trajectors):
        # tranjector
        t1 = np.arange(currentTime, currentTime+pathLen[0])
        t2 = np.arange(t1[-1]+1, t1[-1]+pathLen[1]+1)
        t3 = np.arange(t2[-1]+1, t2[-1]+pathLen[2]+1)
        
        trajectors[t1, path[0][0], path[0][1]] += 1
        trajectors[t2, path[1][0], path[1][1]] += 1
        trajectors[t3, path[2][0], path[2][1]] += 1
        
        if isInterval:
            # subnet
            i = batch_idx*60 + time_idx
            totalLen = sum(pathLen)
            step = (self.endValue-self.startValue)/(totalLen-1)
            steps = np.around(np.arange(self.startValue, self.endValue+step, step), 2)
            self.subLabel[i, path[0][0], path[0][1]] += steps[np.arange(0, pathLen[0])]
            self.counter[i, path[0][0], path[0][1]] += 1
            self.subLabel[i, path[1][0], path[1][1]] += steps[np.arange(pathLen[0], sum(pathLen[:2]))]
            self.counter[i, path[1][0], path[1][1]] += 1
            self.subLabel[i, path[2][0], path[2][1]] += steps[np.arange(sum(pathLen[:2]), sum(pathLen))]
            self.counter[i, path[2][0], path[2][1]] += 1
            # Rnet
            # self.Rfeature[batch_idx, path[0][0], path[0][1], 1] = 1
            # self.Rfeature[batch_idx, path[1][0], path[1][1], 1] = 1
            # self.Rfeature[batch_idx, path[2][0], path[2][1], 1] = 1

        return trajectors


    def image(self):
        trajector = self.trajectors[:10]
        trajector = np.sum(trajector, axis=1)
        nfz = self.NFZ[:10]
        areas = nfz + trajector
        
        # fig, axs = plt.subplots(1, 10, figsize=(40, 6))
        # for ax, title, area in zip(axs, ['trajector', 'subLabel', 'counter', 'Rfeature'], 
        #                                 [trajector, subLabel, counter, Rfeature]):
        for i in range(areas.shape[0]):
            area = areas[i]
            plt.imshow(area, cmap=plt.cm.gnuplot)
            # plt.get_xaxis().set_visible(False)
            # plt.get_yaxis().set_visible(False)
            plt.savefig("img/{0}.png".format(i))


if __name__ == "__main__":
    timeCount = time.time()
    s = SimulatorTest(batch=10, mapSize=100)
    s.generate()
    s.image()
    # print('\ntotal cost {0}'.format(time.time() - timeCount))
    # print("\n--------SubNet--------")
    # print('subTaskList: {0}'.format(s.subTaskList.shape))
    # print('subLabel:    {0}'.format(s.subLabel.shape))
    # print('counter:     {0}'.format(s.counter.shape))
    # print("--------MainNet--------")
    # print('mainTaskList: {0}'.format(s.mainTaskList.shape))
    # print('trajectors:   {0}'.format(s.trajectors.shape))
    # print('subOutput :   {0}'.format(s.subOutput.shape))
    # print('Rfeature:     {0}'.format(s.Rfeature.shape))
    

    # print('----- trajector -----')
    # print(np.max(s.trajectors))
    # print(np.min(s.trajectors))
    # print('----- subOutput -----')
    # print(np.max(s.subOutput))
    # print(np.min(s.subOutput))
    # print('----- Rfeature -----')
    # print(np.max(s.Rfeature))
    # print(np.min(s.Rfeature))





    # np.save('../../../data/zzhao/uav_regression/{0}/{1}.npy'.format('test', 'mainTaskList'), s.mainTaskList)
    # np.save('../../../data/zzhao/uav_regression/{0}/{1}.npy'.format('test', 'trajectors'), s.trajectors)
    # np.save('../../../data/zzhao/uav_regression/{0}/{1}.npy'.format('test', 'Rfeature'), s.Rfeature)

    # np.save('../../../data/zzhao/uav_regression/{0}/{1}.npy'.format('test', 'subTaskList'), s.subTaskList)
    # np.save('../../../data/zzhao/uav_regression/{0}/{1}.npy'.format('test', 'subLabel'), s.subLabel)
    # np.save('../../../data/zzhao/uav_regression/{0}/{1}.npy'.format('test', 'counter'), s.counter)
    

    



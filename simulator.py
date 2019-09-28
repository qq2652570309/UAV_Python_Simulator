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
    def __init__(self, batch = 1, time=120, mapSize=100,  taskNum=15):
        self.batch = batch
        self.map_size = mapSize
        self.time = time
        self.task_num = taskNum
        self.start_time = random.choice(range(50))
        self.area = Area(0,1)
        # In channel, 1st and 2nd are (x, y) launching location, 
        # 3rd and 4th are (x, y) destination location
        # 5th is time
        self.tasks = np.zeros(shape=(batch, 60, taskNum, 5), dtype=int)
        self.trajectors = np.zeros(shape=(batch, 120, mapSize, mapSize), dtype=int)
        self.totalFlyingTime = 0
        self.totalUavNum = 0
        # logging.info('finish init\n')


    def generate(self):
        for batch_idx in range(self.batch):
            startTimeIter = time.time()

            self.area.refresh(mapSize=self.map_size, areaSize=3, num=10)

            # time iteration
            for currentTime in range(self.time):
                
                # task iteration
                startPositions = self.area.getLaunchPoint(n=self.task_num)
                for task_idx, task_val in zip(range(len(startPositions)), startPositions):
                    startRow, startCol, launchingRate = task_val
                # for startRow, startCol, launchingRate in startPositions:
                    startRow = int(startRow)
                    startCol = int(startCol)
                    succ = np.random.uniform(0,1) <= launchingRate
                    
                    # if there is a launching UAV
                    if succ:
                        self.totalUavNum += 1
                        endRow, endCol = self.area.getDestination()

                        # add info into channel
                        if currentTime >= self.start_time + 10 and currentTime < self.start_time + 70:
                            time_idx = currentTime - (self.start_time + 10)
                            # print(time_idx, task_idx, currentTime, self.start_time + 10, self.start_time + 70)
                            self.tasks[batch_idx,time_idx,task_idx,0] = startRow
                            self.tasks[batch_idx,time_idx,task_idx,1] = startCol
                            self.tasks[batch_idx,time_idx,task_idx,2] = endRow
                            self.tasks[batch_idx,time_idx,task_idx,3] = endCol
                            self.tasks[batch_idx,time_idx,task_idx,4] = currentTime

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
                        self.trajectors[batch_idx,t1,startRow,r] += 1
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
                            self.trajectors[batch_idx,t2, c, endCol] += 1
                            self.totalFlyingTime += len(c)
            logging.info('End {0} iteration, cost {1}\n'.format(batch_idx, time.time() - startTimeIter))
            print('End {0} iteration, cost {1}\n'.format(batch_idx, time.time() - startTimeIter))
        self.trajectors = self.trajectors[:,self.start_time:self.start_time+70]

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.disabled = True

    logging.basicConfig(filename='log.txt', format='%(levelname)s:%(message)s', level=logging.INFO)

    logging.info('Started')
    startTimeIter = time.time()
    s = Simulator(batch=1, mapSize=100, time=120)
    s.generate()
    print('UAV Avg Flying Time: ', s.totalFlyingTime/s.totalUavNum)

    logging.info('Finished')
    np.save('data/tasks.npy', s.tasks)
    np.save('data/trajectors.npy', s.trajectors)
    print('Simulation Total Time: ', time.time() - startTimeIter)

'''
# In channel, 0th is status that uav is launching at this second
# 1st is launching rate of this point
# 2nd and 3rd is (x, y) postion of destination point
'''
import time
import os
import random
import logging
import numpy as np
from Area import Area

random.seed(0)
np.random.seed(0)

class Simulator:
<<<<<<< HEAD
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


=======
    def __init__(self, batch = 1, time=200, mapSize=100, taskNum=15, trajectoryTime=70, taskTime=60):
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
        self.Rfeature = np.zeros(shape=(batch, mapSize, mapSize, 2), dtype=np.float32)
        self.totalFlyingTime = 0
        self.totalUavNum = 0
        if os.path.exists('./log.txt'):
            os.remove('log.txt')

    
>>>>>>> pnet_time
    def generate(self):
        for batch_idx in range(self.batch):
            startTimeIter = time.time()
            trajectors = np.zeros(shape=(self.time, self.map_size, self.map_size), dtype=int)

            self.area.refresh(mapSize=self.map_size, areaSize=3, num=10)
            self.drawPatten(batch_idx)
            start_time = random.choice(range(60, 80))

            # time iteration
            for currentTime in range(self.time):
                
                # if currentTime == start_time + 30:
                #     self.area.updateLaunchingRate()
                
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
                        self.Rfeature[batch_idx, startRow, startCol, 0] = launchingRate
                        self.Rfeature[batch_idx, endRow, endCol, 0] = 0.3

<<<<<<< HEAD
            startPositions, endPositions = self.generateLaunchingPoints()
            # startPoints = self.choosePoints(self.startPointsNum)
            # startPositions = list(map(lambda x: (x//self.column, x%self.column), startPoints))
            # endPoints = self.choosePoints(self.endPointsNum)
            # endPositions = list(map(lambda x: (x//self.column, x%self.column), endPoints))
            
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
                        np.random.shuffle(endPositions)
                        endRow, endCol  = random.choice(endPositions)
=======
                        # add info into channel
                        if currentTime >= start_time + 10 and currentTime < start_time + self.trajectoryTime:
                            time_idx = currentTime - (start_time + 10)
                            # print(time_idx, task_idx, currentTime, self.start_time + 10, self.start_time + 70)
                            self.tasks[batch_idx,time_idx,task_idx,0] = startRow
                            self.tasks[batch_idx,time_idx,task_idx,1] = startCol
                            self.tasks[batch_idx,time_idx,task_idx,2] = endRow
                            self.tasks[batch_idx,time_idx,task_idx,3] = endCol
                            self.tasks[batch_idx,time_idx,task_idx,4] = currentTime

>>>>>>> pnet_time
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
<<<<<<< HEAD
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
        np.random.shuffle(launchingArea)
        np.random.shuffle(destination)
        return launchingArea, destination


logger = logging.getLogger()
logger.disabled = True

logging.basicConfig(filename='log.txt', format='%(levelname)s:%(message)s', level=logging.INFO)

logging.info('Started')
startTimeIter = time.time()
# s = Simulator(iteration=2, row=4, column=4, time=5, startPointsNum=3, endPointsNum=3)
# s = Simulator(iteration=10000, row=16, column=16, time=60, startPointsNum=15, endPointsNum=15)
s = Simulator(iteration=100, row=32, column=32, time=120, startPointsNum=4, endPointsNum=4)
s.generate()
# s.dataProcess()
logging.info('Finished')
np.save('data/trainingSets_raw.npy', s.trainingSets)
np.save('data/groundTruths_raw.npy', s.groundTruths)
print('total time: ', time.time() - startTimeIter)
# logging.info('trainingSets: \n{0}'.format(s.trainingSets))
# logging.info('groundTruths: \n{0}'.format(s.groundTruths))
=======
                            trajectors[t2, c, endCol] += 1
                            self.totalFlyingTime += len(c)
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
                self.Rfeature[batch_idx, startRow, r, 1] = 1
                if startRow < endRow:
                    c = np.arange(startRow+1, endRow+1)
                else:
                    c = np.arange(endRow, startRow)[::-1]
                self.Rfeature[batch_idx, c, endCol, 1] = 1
                
                    

if __name__ == "__main__":
    s = Simulator(batch=1, mapSize=100, time=120)
    
    logger = logging.getLogger()
    logger.disabled = True
    
    logging.basicConfig(filename='log.txt', format='%(levelname)s:%(message)s', level=logging.INFO)

    logging.info('Started')
    startTimeIter = time.time()
    
    # s.generate()
    # print('UAV Avg Flying Time: ', s.totalFlyingTime/s.totalUavNum)

    logging.info('Finished')
    np.save('data/tasks.npy', s.tasks)
    np.save('data/trajectors.npy', s.trajectors)
    # print('Simulation Total Time: ', time.time() - startTimeIter)
>>>>>>> pnet_time

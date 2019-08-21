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
    def __init__(self, iteration = 1, time=60, row=0, column=0,  startPointsNum=10, endPointsNum=10, requiredDist=15):
        self.iteration = iteration
        self.row = row
        self.column = column
        self.time = time
        self.requiredDist = requiredDist
        self.startPointsNum = startPointsNum
        self.endPointsNum = endPointsNum
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

            startPositions, endPositions = self.generatePositions()
            self.setColor(index, startPositions, endPositions)
            # startPoints = self.choosePoints(self.startPointsNum)
            # startPositions = list(map(lambda x: (x//self.column, x%self.column), startPoints))
            # endPoints = self.choosePoints(self.endPointsNum)
            # endPositions = list(map(lambda x: (x//self.column, x%self.column), endPoints))

            for startRow, startCol in startPositions:
                logging.info('   At start Point ({0}, {1})'.format(startRow, startCol))
                # set traning sets
                startRow = int(startRow)
                startCol = int(startCol)
                self.trainingSets[index,:,startRow,startCol,1] = np.random.uniform(0, 1)
                # generate ground truth
                for currentTime in range(self.time):
                    succ = np.random.uniform(0,1) <= self.trainingSets[index,currentTime,startRow,startCol,1]
                    if succ:
                        np.random.shuffle(endPositions)
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

    def generatePositions(self):
        startPoints = self.choosePoints(self.startPointsNum)
        startPositions = list(map(lambda x: (x//self.column, x%self.column), startPoints))
        endPositions = []
        endPoints = np.arange(self.row*self.column, dtype=np.int)
        while len(endPositions) < self.endPointsNum :
            indexEnd = np.random.randint(low=0, high=(len(endPoints)))
            endPosition = (endPoints[indexEnd]//self.column, endPoints[indexEnd]%self.column)
            endPoints = np.delete(endPoints, indexEnd)
            if self.computeDistance(endPosition, startPositions, self.requiredDist):
                endPositions.append(endPosition)
        return startPositions, endPositions

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


logger = logging.getLogger()
logger.disabled = True

logging.basicConfig(filename='log.txt', format='%(levelname)s:%(message)s', level=logging.INFO)

logging.info('Started')
startTimeIter = time.time()
# s = Simulator(iteration=2, row=4, column=4, time=5, startPointsNum=3, endPointsNum=3)
# s = Simulator(iteration=10000, row=16, column=16, time=60, startPointsNum=15, endPointsNum=15)
s = Simulator(iteration=10000, row=32, column=32, time=120, startPointsNum=12, endPointsNum=12, requiredDist=5)
s.generate()
# s.dataProcess()
logging.info('Finished')
np.save('data/trainingSets_raw.npy', s.trainingSets)
np.save('data/groundTruths_raw.npy', s.groundTruths)
np.save('data/positions.npy', s.positions)
print('total time: ', time.time() - startTimeIter)
# logging.info('trainingSets: \n{0}'.format(s.trainingSets))
# logging.info('groundTruths: \n{0}'.format(s.groundTruths))
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
    def __init__(self, batch = 1, time=60, row=0, column=0,  taskNum=10, timeInterval=5):
        self.batch = batch
        self.row = row
        self.column = column
        self.time = time
        self.taskNum = taskNum
        self.timeInterval = timeInterval
        self.area = Area(0, 1)
        # In channel, 1st and 2nd are (x, y) launching location, 3rd and 4th are (x, y) destination location
        self.trainingSets = np.zeros(shape=(self.batch, self.time, self.row, self.column, 1), dtype=np.float32)
        self.groundTruths = np.zeros(shape=(self.batch, self.time, self.row, self.column), dtype=np.float32)
        self.totalFlyingTime = 0
        self.totalUavNum = 0


    def generate(self):
        startTimeTotal = time.time()
        
        for batch_idx in range(self.batch):
            startTimeIter = time.time()
        
            self.area.refresh(mapSize=self.row, areaSize=3, num=10)

            for currentTime in range(self.time):
                startPositions = self.area.getLaunchPoint()

                # generate ground truth
                for startRow, startCol, launchingRate in startPositions:
                    startRow = int(startRow)
                    startCol = int(startCol)

                    succ = np.random.uniform(0,1) <= launchingRate
                    if succ:
                        self.totalUavNum += 1
                        self.trainingSets[batch_idx,currentTime,startRow,startCol,1] = launchingRate

        print(startPositions)



s = Simulator(row=100, column=100)
s.generate()


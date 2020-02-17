import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time as showTime

class PostPorcess:
    def __init__(self):
        self.batch = 3000
        time=200
        mapSize=100
        taskNum=15
        trajectoryTime=70
        taskTime=60
        self.startValue = 0.25
        self.endValue = 0.75
        # channels: mainTaskList[0] and mainTaskList[1] is launching location
        # channels: mainTaskList[2] and mainTaskList[3] is landing location 
        # channels: mainTaskList[5] is launching time
        # mainTaskList = (3000, 60, 15, 5)
        self.mainTaskList = np.zeros(shape=(self.batch, taskTime, taskNum, 5), dtype=int)
        # every timestep, number of uav on each grid
        # used for generate density map (label) and init density (input)
        self.trajectors = np.zeros(shape=(self.batch, trajectoryTime, mapSize, mapSize), dtype=int)
        # subOutput = (3000, 60, 100, 100), tasklist as input for MainNet
        self.subOutput = np.zeros(shape=(self.batch, taskTime, mapSize, mapSize), dtype=float)
        self.counter = np.zeros(shape=(self.batch, taskTime, mapSize, mapSize), dtype=int)

        self.maxRow = -2**8
        self.maxCol = -2**8
        
        self.trajectorOver = np.array(np.zeros(3001), dtype=bool)
        self.mainTaskListOver = np.array(np.zeros(3001), dtype=bool)
        
    def generateTrajectors(self):
        for batch in range(2500, self.batch):
            startTimeTotal = showTime.time()
            trajectorFileName = "../../../data/zzhao/uav_regression/csvData/batch_{0}/uav_index.csv".format(batch)
            tdata = pd.read_csv(trajectorFileName, usecols=['Time_Step', 'UAV_ID', 'X_Distance', 'Y_Distance'])
            timeRange = np.arange(230, 300, 1) # time interval from 230 to 299
            intervalPartT = tdata[tdata['Time_Step'].isin(timeRange)]
            
            # trajector part
            for index, row in intervalPartT.iterrows():
                time = int(row['Time_Step'])-230
                id = row['UAV_ID']
                colPos = int(round(row['X_Distance']))
                rowPos = int(round(row['Y_Distance']))
                # print("time={0}, id={1}, row={2}, col={3}".format(time, id, rowPos,colPos))

                if colPos >= 100 or rowPos >= 100 :
                    print("trajector overflow: {0}".format(batch))
                    self.trajectorOver[batch] = True
                    self.trajectors[batch] = np.zeros(shape=(70, 100, 100), dtype=int)
                    break
                else :
                    self.trajectors[batch, time, rowPos, colPos] += 1

            missionFileName = "../../../data/zzhao/uav_regression/csvData/batch_{0}/mission_report.csv".format(batch)
            mdata = pd.read_csv(missionFileName, usecols=['Start_Time', 'Start_Location', 'End_Location'])
            timeRange = np.arange(240, 300, 1) # time interval from 230 to 299
            intervalPartM = mdata[mdata['Start_Time'].isin(timeRange)]
            taskPositions = np.zeros(shape=(60), dtype=int)

            # task list part
            for index, row in intervalPartM.iterrows():
                time = int(row['Start_Time'])-240
                startCol, startRow = row['Start_Location'][1:-1].split(':')
                endCol, endRow = row['End_Location'][1:-1].split(':')
                startCol = round(float(startCol))
                startRow = round(float(startRow))
                endCol = round(float(endCol))
                endRow = round(float(endRow))
                # print('({0}, {1}) ==> ({2}, {3})'.format(startRow, startCol, endRow, endCol))
                self.mainTaskList[batch, time, taskPositions[time], 0] = startRow
                self.mainTaskList[batch, time, taskPositions[time], 1] = startCol
                self.mainTaskList[batch, time, taskPositions[time], 2] = endRow
                self.mainTaskList[batch, time, taskPositions[time], 3] = endCol
                self.mainTaskList[batch, time, taskPositions[time], 4] = int(row['Start_Time'])
                taskPositions[time] += 1

                if self.trajectorOver[batch] :
                    break
                
                if startRow >= 100 or startCol >= 100 or endRow >= 100 or endCol >= 100 :
                    print("mainTaskList overflow: {0}".format(batch))
                    self.mainTaskListOver[batch] = True
                    self.mainTaskList[batch] = np.zeros(shape=(60, 15, 5), dtype=int)
                    self.subOutput[batch] = np.zeros(shape=(60, 100, 100), dtype=float)
                    self.counter[batch] = np.zeros(shape=(60, 100, 100), dtype=int)
                    break
                else :
                    # subnet part
                    self.subnetFlow(sr=startRow, sc=startCol, er=endRow, ec=endCol, batch=batch, time=time)
            self.subOutput[batch] = np.nan_to_num(self.subOutput[batch] / self.counter[batch])
            print("batch_{0}, {1} \n".format(batch, showTime.time() - startTimeTotal))


    # generate flow subnet
    def subnetFlow(self, sr, sc, er, ec, batch, time):
        startRow, startCol = sr, sc
        endRow, endCol = er, ec
        
        rowDistance = abs(startRow-endRow)+1
        colDistance = abs(startCol-endCol)+1
        
        # compute each step value
        pathLen = 0
        if rowDistance == 1 :
            pathLen = colDistance
        elif colDistance == 1 :
            pathLen = rowDistance
        else:
            pathLen = max(rowDistance, colDistance)-1
        step = (self.endValue-self.startValue)/pathLen
        steps = np.around(np.arange(start=self.startValue, stop=self.endValue+step, step=step), 2)

        if rowDistance <= colDistance:
            rPos = np.arange(min(endRow, startRow), max(endRow, startRow)+1)
            if startRow > endRow :
                rPos=rPos[::-1]
            
            cPosList = np.arange(min(startCol, endCol), max(startCol, endCol)+1)
            if startCol > endCol :
                cPosList=cPosList[::-1]
            cPosList = np.array_split(cPosList, len(rPos))

            spIndex = 0
            for rp, cPos in zip(rPos, cPosList) :
                self.subOutput[batch, time, rp, cPos] += steps[spIndex:spIndex+len(cPos)]
                self.counter[batch, time, rp, cPos] += 1
                spIndex += len(cPos)
        
        else:
            cPos = np.arange(min(endCol, startCol), max(endCol, startCol)+1)
            if startCol > endCol :
                cPos=cPos[::-1]
            
            rPosList = np.arange(min(startRow, endRow), max(startRow, endRow)+1)
            if startRow > endRow :
                rPosList=rPosList[::-1]
            rPosList = np.array_split(rPosList, len(cPos))

            spIndex = 0
            for cp, rPos in zip(cPos, rPosList) :
                self.subOutput[batch, time, rPos, cp] += steps[spIndex:spIndex+len(rPos)]
                self.counter[batch, time, rPos, cp] += 1
                spIndex += len(rPos)
        

    def testTrajector(self):
        # filter data out off interval
        timeRange = np.arange(240, 300, 1) # time interval from 240 to 299
        for batch in range(1, self.batch+1, 1):
            trajectorFileName = "uav_index.csv".format(batch)
            tdata = pd.read_csv(trajectorFileName, usecols=['Time_Step', 'UAV_ID', 'X_Distance', 'Y_Distance'])
            intervalPartT = tdata[tdata['Time_Step'].isin(timeRange)]
            
            tMin = 1024
            tMax = -1024

            for index, row in intervalPartT.iterrows():
                time = int(row['Time_Step'])
                id = row['UAV_ID']
                colPos = round(row['X_Distance'])
                rowPos = round(row['Y_Distance'])
                # print("time={0}, id={1}, row={2}, col={3}".format(time, id, rowPos,colPos))
                # self.trajectors[batch, time, rowPos, colPos] += 1
                if tMin > time:
                    tMin = time
                if tMax < time:
                    tMax = time
            print("tMin: ", tMin)
            print("tMax: ", tMax)

    def testMission(self):
        timeRange = np.arange(280, 300, 1) # time interval from 240 to 299
        for batch in range(1, self.batch+1, 1):
            missionFileName = "mission_report.csv".format(batch)
            mdata = pd.read_csv(missionFileName, usecols=['Start_Time', 'Start_Location', 'End_Location'])
            intervalPartM = mdata[mdata['Start_Time'].isin(timeRange)]

            tMin = 1024
            tMax = -1024

            for index, row in intervalPartM.iterrows():
                time = int(row['Start_Time'])
                startCol, startRow = row['Start_Location'][1:-1].split(':')
                endCol, endRow = row['End_Location'][1:-1].split(':')
                startCol = round(float(startCol))
                startRow = round(float(startRow))
                endCol = round(float(endCol))
                endRow = round(float(endRow))
                print('({4}: {0}, {1}) ==> ({2}, {3})'.format(startRow, startCol, endRow, endCol, time))
                if tMin > time:
                    tMin = time
                if tMax < time:
                    tMax = time
            # print("tMin: ", tMin)
            # print("tMax: ", tMax)

    def image(self):
        areas = self.trajectors[0]
        for i in range(areas.shape[0]):
            area = areas[i]
            
            plt.imshow(area, cmap=plt.cm.gnuplot)
            # plt.get_xaxis().set_visible(False)
            # plt.get_yaxis().set_visible(False)
            plt.savefig("img/test_{0}.png".format(i))


    # (30, 32, 32) --> (32, 32)
    def generateDensity(self, gtr):
        temp = np.sum(gtr, axis=1)
        return temp

    # lumped map divided time, return with batch normalization
    def averageDensity(self, gtr, time):
        gtr = gtr/time
        return gtr
    
    # generate density map from timestep start -> end
    def intervalDensity(self, data, start, end):
        interval = data[:,start:end]
        densityMap = self.generateDensity(interval)
        return self.averageDensity(densityMap, end-start)

    def save(self, data, name='test', directory='test', subDirectory='subtest'):
        if not os.path.exists('../../../data/zzhao/uav_regression/{0}'.format(directory)):
            os.mkdir('../../../data/zzhao/uav_regression/{0}'.format(directory))
            os.chmod('../../../data/zzhao/uav_regression/{0}'.format(directory), 0o777)
        if not os.path.exists('../../../data/zzhao/uav_regression/{0}/{1}'.format(directory, subDirectory)):
            os.mkdir('../../../data/zzhao/uav_regression/{0}/{1}'.format(directory, subDirectory))
            os.chmod('../../../data/zzhao/uav_regression/{0}/{1}'.format(directory, subDirectory), 0o777)
        
        if os.path.exists('../../../data/zzhao/uav_regression/{0}/{1}/{2}.npy'.format(directory, subDirectory, name)):
            os.remove('../../../data/zzhao/uav_regression/{0}/{1}/{2}.npy'.format(directory, subDirectory, name))

        np.save('../../../data/zzhao/uav_regression/{0}/{1}/{2}.npy'.format(directory, subDirectory, name), data)
        os.chmod('../../../data/zzhao/uav_regression/{0}/{1}/{2}.npy'.format(directory, subDirectory, name), 0o777)
        print(' {0}/{1}: {2} save complete\n'.format(subDirectory, name, data.shape))

    def featureLabel(self, directory='test'):
        # ---------------------- main network ----------------------      
        trajectory = np.copy(self.trajectors)
        # densityLabel = (batch, 100, 100) in last 10 timesteps
        densityLabel = self.intervalDensity(trajectory, trajectory.shape[1]-10, trajectory.shape[1])
        self.save(densityLabel, name='label_mainnet', directory=directory, subDirectory='fushion')

        # desntiyFeature = (batch, 100, 100) in first 10 timesteps
        desntiyFeature = self.intervalDensity(trajectory, 0, 10)
        self.save(desntiyFeature, name='data_init', directory=directory, subDirectory='fushion')
        # tasklist = (batch, 60, 15, 5)
        self.save(self.mainTaskList, name='data_tasks', directory=directory, subDirectory='fushion')
        # subnet output = (batch, 60, 100, 100)
        self.save(self.subOutput, name='data_subnet', directory=directory, subDirectory='fushion')
        # subnet Cube output = (batch, 60, 100, 100)
        # self.save(self.subOutputCube, name='data_subnet_cube', directory=directory, subDirectory='fushion')

        print('finish saving')


if __name__=="__main__":
    pp = PostPorcess()
    # pp.testTrajector()
    # pp.testMission()
    pp.generateTrajectors()·
    # pp.featureLabel(directory='javaP2P')
    # pp.image()

    print(np.min(pp.trajectors))
    print(np.max(pp.trajectors))
    


import numpy as np
import logging
import time
from simulator import Simulator
from dataPreprocess import Preprocess
from generateImage import Image

# nomalize (100, 100)
def batchNormalize(gtr):
    gtr = (gtr - np.min(gtr)) / (np.max(gtr) - np.min(gtr))
    return gtr

def generateDensity(gtr):
    temp = np.sum(gtr, axis=0)
    return temp 

# lumped map divided time, return with batch normalization
def averageDensity(gtr, time):
    gtr = gtr/time
    return batchNormalize(gtr)


if __name__ == "__main__":
    simulateTime = 200
    batchSize = 10
    moveStep = 10
    window = 20
    
    
    s = Simulator(batch=batchSize, time=simulateTime)
    startTimeTotal = time.time()
    s.generate()

    print('input shape is {0}\n'.format(s.trajectors.shape))
    
    stable = s.trajectors[:, 150:200, :]
    # stableMap = np.zeros(shape=())
    end = simulateTime-window
    # batchSize = s.trajectors.shape[0]
    intervalNum = int(end/moveStep) + 1

    analysis = np.zeros(shape=(batchSize, intervalNum, 100, 100))
    print('analysis shape is {0}\n'.format(analysis.shape))
    

    for b in range(len(s.trajectors)):
        for i in range(0, end+1, moveStep):
            intervalTrajectory = s.trajectors[b, i:(i+window), :]
            intervalDensity = generateDensity(intervalTrajectory)
            intervalDensity = intervalDensity/window
            intervalDensity = averageDensity(intervalDensity,window)
            analysis[b, int( (i+(moveStep-1)) / moveStep )] = intervalDensity
            # print(i, ' : ', int( (i+(moveStep-1)) / moveStep))
    
    '''
    for b in range(len(s.trajectors)):
        for i in range(0, end+1, interval):
            intervalTrajectory = s.trajectors[b, i:(i+interval), :]
            intervalDensity = generateDensity(intervalTrajectory)
            intervalDensity = intervalDensity/interval
            intervalDensity = averageDensity(intervalDensity,interval)
            analysis[b, int( (i+(interval-1)) / interval )] = intervalDensity
    #'''
    
    
    mae = np.zeros((batchSize,intervalNum-1))
    # delta_mae = np.zeros((batchSize,18))
    for b in range(batchSize):
        print('sample {0} :'.format(b))
        for i in range(intervalNum-1):
            mae1 = (np.abs(analysis[b, i] - analysis[b, intervalNum-1])).mean(axis=None) * 100
            # mae2 = (np.abs(analysis[b, i+1] - analysis[b, intervalNum-1])).mean(axis=None) * 100
            # mae = np.sum(np.abs(analysis[i] - analysis[10]), axis=None)
            # print('     ', abs(mae1-mae2))
            # delta = abs(mae1-mae2)
            print('     ', abs(mae1))
            mae[b,i] = mae1
            # delta_mae[b,i] = delta
            
    # print(delta_mae.shape)
    print(mae.shape)
    
    np.save('data/mae.npy', mae)
    # np.save('data/delta_mae.npy', delta_mae)
    
    
    

    
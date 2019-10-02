import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from simulator import Simulator
from dataPreprocess import Preprocess

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


def getMAE():
    simulateTime = 200
    batchSize = 20
    moveStep = 10
    window = 10
    
    
    s = Simulator(batch=batchSize, time=simulateTime)
    s.generate()

    print('------Analysis----------')
    print('input shape is {0}'.format(s.trajectors.shape))
    print('moveStep={0}'.format(moveStep))
    print('window={0}\n'.format(window))

    end = simulateTime-window
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
    
    mae = np.zeros((batchSize-1,intervalNum))

    for i in range(intervalNum):
        for b in range(batchSize-1):
            mae1 = (np.abs(analysis[b, i] - analysis[b+1, i])).mean(axis=None) * 100
            # print('     ', abs(mae1))
            mae[b,i] = mae1
    
    return mae




allMAE = getMAE()
print(allMAE.shape)


plt.figure(figsize=(50, 4))
n = int(allMAE.shape[0]/10)


for index in range(n+1):
    ax = plt.subplot(1, n+1, index+1)
    if index==n:
        mae = allMAE[index*10:(index*10+9)]
    else:
        mae = allMAE[index*10:(index*10+10)]
    for m in mae:
        ax.plot(range(mae.shape[1]), m)
    ax.set(xlabel='time', ylabel='mae')
    ax.grid()


plt.savefig("img/window10.png")






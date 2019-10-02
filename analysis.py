import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
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
    batchSize = 50
    moveStep = 10
    window = 10
    
    
    s = Simulator(batch=batchSize, time=simulateTime)
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
    
    mae = np.zeros((batchSize,intervalNum-1))
    if window == 10:
        mae = np.zeros((batchSize,intervalNum))
    # delta_mae = np.zeros((batchSize,18))
    for b in range(batchSize):
        print('sample {0} :'.format(b))
        n = intervalNum - 1
        if window == 10:
            n = intervalNum
        for i in range(n):
            mae1 = (np.abs(analysis[b, i] - analysis[b, intervalNum-1])).mean(axis=None) * 100
            # mae2 = (np.abs(analysis[b, i+1] - analysis[b, intervalNum-1])).mean(axis=None) * 100
            # mae = np.sum(np.abs(analysis[i] - analysis[10]), axis=None)
            # print('     ', abs(mae1-mae2))
            # delta = abs(mae1-mae2)
            print('     ', abs(mae1))
            mae[b,i] = mae1
            # delta_mae[b,i] = delta
        mae[b] = batchNormalize(mae[b])
    return mae





# time = [0,1,2,3,4,5,6]
# mae = np.round(np.random.rand(7), 2)


# p = np.percentile(mae, 90)

# print(mae)

allMAE = getMAE()
# allMAE = allMAE[:,:-2]
print(allMAE.shape)

plt.figure(figsize=(50, 4))
n = int(allMAE.shape[0]/10)
for index in range(n):
    ax = plt.subplot(1, n, index+1)
    mae = allMAE[index*10:(index*10+10)]
    for m in mae:
        ax.plot(range(mae.shape[1]), m)

    ax.set(xlabel='time', ylabel='mae')
    ax.grid()
    


plt.savefig("img/window10.png")






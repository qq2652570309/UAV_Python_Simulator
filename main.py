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
    s = Simulator()
    startTimeTotal = time.time()
    s.generate()

    print('input shape is {0}\n'.format(s.trajectors.shape))


    end = 100
    interval=10
    batch = int(end/interval) + 1

    analysis = np.zeros(shape=(batch, 100, 100))
    print('analysis shape is {0}\n'.format(analysis.shape))

    for b in range(len(s.trajectors)):
        for i in range(0, end+1, interval):
            intervalTrajectory = s.trajectors[b, i:(i+interval),:]
            intervalDensity = generateDensity(intervalTrajectory)
            intervalDensity = averageDensity(intervalDensity,interval)
            analysis[int((i+9)/10)] = intervalDensity

    for i in range(10):
        mae = (np.abs(analysis[i] - analysis[10])).mean(axis=None) * 100
        # mae = np.sum(np.abs(analysis[i] - analysis[10]), axis=None)
        print(mae)


    
    data = [analysis]
    rowHeader = ['analysis']

    i = Image(data, rowHeader, 'intervel')
    i.generate()
    #'''
    '''
    A = np.array([
        [0.1 , 0.5],
        [0.7, 0.9],
    ])

    B = np.array([
        [0.2 , 0.5],
        [0.7, 0.4],
    ])

    print((A - B))
    print((A - B)**2)

    mse = ((A - B)**2).mean(axis=None)
    print(mse)
    '''
    
from simulator import Simulator
from dataPreprocess import Preprocess
from model import Lstm_Cnn_Model
from generateImage import Image
from Area import Area

import sys
import numpy as np
import time

# print ('Number of arguments:', len(sys.argv), 'arguments.')
# print ('Argument List:', str(sys.argv))


def simulate(n=None, uavNum=None):
    if n==None:
        s = Simulator(iteration=10000, row=32, column=32, time=120, uavNum=uavNum, timeInterval=5)
    else:
        s = Simulator(iteration=n, row=32, column=32, time=120, uavNum=uavNum, timeInterval=5)
    startTimeIter = time.time()
    s.generate()
    print('trainingSets_raw shape: ', s.trainingSets.shape)
    print('groundTruths_raw shape: ', s.groundTruths.shape)
    np.save('data/trainingSets_raw.npy', s.trainingSets)
    np.save('data/groundTruths_raw.npy', s.groundTruths)
    # np.save('data/positions.npy', s.positions)
    print('total time: ', time.time() - startTimeIter)

def processSequence(p, mode):
    p.splitByTime(20)
    if mode == 'trajectory':
        p.oneOrZero()
        p.computeWeights()
    else:
        p.generateDensity()
        p.batchNormalize()
    p.checkGroundTruthIdentical()
    p.averageLaunchingNumber()
    return p.tsr, p.gtr

def preprocess(mode='density'):
    p = Preprocess()
    processSequence(p, mode=mode)
    if mode=='density':
        p.saveData(name='density')
    if mode=='trajectory':
        p.saveData(name='trajectory')


def train(mode='density', epics=3, weight=1):
    CSM = Lstm_Cnn_Model(epics=epics, weight=weight)
    if mode=='density':
        CSM.loadData(
            "data/trainingSets_density.npy",
            "data/groundTruths_density.npy"
        )
        CSM.layers()
        CSM.train('mse')
    elif mode=='trajectory':
        CSM.loadData(
            "data/trainingSets_trajectory.npy",
            "data/groundTruths_trajectory.npy"
        )
        CSM.lstmLayers()
        CSM.train('recall')


def img(mode='density', ckpt=''):
    s = Simulator(iteration=10, row=32, column=32, time=120, uavNum=2, timeInterval=5)
    s.generate()
    
    p = Preprocess(groundTruth=s.groundTruths, trainingSets=s.trainingSets)
    x, y = processSequence(p,mode)

    
    CSM = Lstm_Cnn_Model()
    if mode is 'density':
        CSM.layers()
        prediction, groundtruth = CSM.imageData(
            ckpt=ckpt,
            x=x,
            y=y,
            mode='mse')
    else:
        CSM.lstmLayers()
        prediction, groundtruth = CSM.imageData(
            ckpt=ckpt,
            x=x,
            y=y,
            mode='recall',
            isRound=True)
    print(prediction.shape)
    print(groundtruth.shape)


    data=[groundtruth, prediction, s.positions]
    rowHeader = ['groundTrue', 'prediction', 'positions']
    if mode is 'density':
        colHeader = 'sample'
    else:
        colHeader = 'time'
    i = Image(data,rowHeader,colHeader)
    i.generate()
    

def main():
    mode='density'
    # mode='trajectory'

    # simulate(n=10000, uavNum=10)
    # preprocess(mode=mode)
    train(mode=mode, epics=3, weight=151.97)
    # img(mode=mode, ckpt='checkpoints/mse-02-0.06')



if __name__ == "__main__":
    main()
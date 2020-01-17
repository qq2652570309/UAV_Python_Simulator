import numpy as np
import os
import random
from copy import deepcopy
from sys import getsizeof

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time

random.seed(0)
np.random.seed(0)

def batchNormalize(gtr):
        if np.max(gtr) != 0:
            gtr = (gtr - np.min(gtr)) / (np.max(gtr) - np.min(gtr))
        # logging.info('      after batchNormalize')
        print('          min: {0}'.format(np.min(gtr)))
        print('          max: {0}'.format(np.max(gtr)))
        print('          mean: {0}'.format(np.mean(gtr)))
        print('          median: {0}'.format(np.median(gtr)))
        return gtr

def averageDensity(gtr, time):
        tmp = gtr/time
        return batchNormalize(tmp)

def generateDensity(gtr):
        temp = np.sum(gtr, axis=1)
        return temp   

def intervalDensity(data, start, end):
    interval = data[:,start:end]
    densityMap = generateDensity(interval)
    return averageDensity(densityMap, end-start)

def intervalDensity1(data, start, end):
    interval = data[:,start:end]
    densityMap = generateDensity(interval)
    # return averageDensity(densityMap, end-start)
    return densityMap

##############################
# only load without modify
##############################


# label_mainnet = np.load('../../../data/zzhao/uav_regression/javaP2P/fushion/label_mainnet.npy', mmap_mode='r')
# print(label_mainnet.shape)
# data_init = np.load('../../../data/zzhao/uav_regression/javaP2P/fushion/data_init.npy', mmap_mode='r')
# data_subnet = np.load('../../../data/zzhao/uav_regression/javaP2P/fushion/data_subnet.npy', mmap_mode='r')
# print(data_subnet.shape)
# data_subnet_cube = np.load('../../../data/zzhao/uav_regression/astarProbility/fushion/data_subnet_cube.npy', mmap_mode='r')

prediction = np.load('data/sixSeg/pred_ones.npy', mmap_mode='r')
label = np.load('data/sixSeg/label_sixSeg.npy', mmap_mode='r')
init = np.load('data/sixSeg/init_sixSeg.npy', mmap_mode='r')


# print("min subnet: ", np.min(data_subnet[data_subnet>0]))
# print("max subnet: ", np.max(data_subnet))

# print("min subnet_cube: ", np.min(data_subnet_cube))
# print("max subnet_cube: ", np.max(data_subnet_cube))

# print("min init: ", np.min(data_init))
# print("max init: ", np.max(data_init))

# print("min mainnet: ", np.min(label_mainnet))
# print("max mainnet: ", np.max(label_mainnet))


matrixImg0 = prediction
matrixImg1 = label
matrixImg2 = init
# print(matrixImg.shape)

# values = np.unique(label_mainnet)
# print(values)


'''
#####################
# load and modify
#####################

# label_mainnet = np.load('../../../data/zzhao/uav_regression/mainFlow/fushion/label_mainnet.npy')
# data_tasks = np.load('../../../data/zzhao/uav_regression/mainFlow/fushion/data_tasks.npy')
# data_subnet = np.load('../../../data/zzhao/uav_regression/mainFlow/fushion/data_subnet.npy')
# data_init = np.load('../../../data/zzhao/uav_regression/mainFlow/fushion/data_init.npy')
'''

#################
# visualization
#################

plt.figure(figsize=(12, 6))

ax = plt.subplot(3, 5, 1)
plt.imshow(matrixImg0[0])
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(True)
ax.set_ylabel("prediction", rotation=90, size='large')
plt.gray()

ax = plt.subplot(3, 5, 2)
plt.imshow(matrixImg0[1])
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.gray()

ax = plt.subplot(3, 5, 3)
plt.imshow(matrixImg0[2])
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.gray()

ax = plt.subplot(3, 5, 4)
plt.imshow(matrixImg0[3])
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.gray()

ax = plt.subplot(3, 5, 5)
plt.imshow(matrixImg0[4])
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.gray()

ax = plt.subplot(3, 5, 6)
plt.imshow(matrixImg1[0])
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(True)
ax.set_ylabel("label", rotation=90, size='large')
plt.gray()

ax = plt.subplot(3, 5, 7)
plt.imshow(matrixImg1[1])
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.gray()

ax = plt.subplot(3, 5, 8)
plt.imshow(matrixImg1[2])
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.gray()

ax = plt.subplot(3, 5, 9)
plt.imshow(matrixImg1[3])
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.gray()

ax = plt.subplot(3, 5, 10)
plt.imshow(matrixImg1[4])
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.gray()

ax = plt.subplot(3, 5, 11)
plt.imshow(matrixImg2[0][0])
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(True)
ax.set_ylabel("init", rotation=90, size='large')
plt.gray()

ax = plt.subplot(3, 5, 12)
plt.imshow(matrixImg2[1][0])
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.gray()

ax = plt.subplot(3, 5, 13)
plt.imshow(matrixImg2[2][0])
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.gray()

ax = plt.subplot(3, 5, 14)
plt.imshow(matrixImg2[3][0])
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.gray()

ax = plt.subplot(3, 5, 15)
plt.imshow(matrixImg2[4][0])
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.gray()



plt.suptitle("SixSeg Ones")

plt.savefig("visulization/sixSeg/SixSeg_Ones.png")

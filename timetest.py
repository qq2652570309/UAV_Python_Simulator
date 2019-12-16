import numpy as np
import os
import time


A = np.zeros((5000,5000))
startTimeIter = time.time()
for x in range(5000):
    for y in range(5000):
        A[x,y] = 1;
print(time.time() - startTimeIter)

B = np.zeros((5000,5000))
startTimeIter = time.time()
for x in range(5000):
    B[x,range(0,5000)] = 1
print(time.time() - startTimeIter)

C = np.zeros((5000,5000))
startTimeIter = time.time()
for x in range(5000):
    C[x,np.arange(0,5000)] = 1
print(time.time() - startTimeIter)

D = np.zeros((5000,5000))
startTimeIter = time.time()
for x in np.arange(5000):
    for y in np.arange(5000):
        D[x,y] = 1;
print(time.time() - startTimeIter)

E = np.zeros((5000,5000))
startTimeIter = time.time()
E[0:5000,0:5000] = 1
print(time.time() - startTimeIter)

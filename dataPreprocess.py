import numpy as np


class Preprocess:

    def __init__(self):
        self.gtr = np.load("data/groundTruths_raw.npy")
        self.tsr = np.load("data/trainingSets_raw.npy")
        print('raw trainingSets', self.tsr.shape)
        print('raw groundTruth: ', self.gtr.shape)

    def split(self):
        dgtr = self.gtr[:,-1:]
        dgtr = dgtr.reshape((dgtr.shape[0],dgtr.shape[2],dgtr.shape[3]))
        for i in range(dgtr.shape[0]):
            dgtr[i] = (dgtr[i] - np.min(dgtr[i])) / (np.max(dgtr[i]) - np.min(dgtr[i]))
            if np.max(dgtr[i]) != 1 or np.min(dgtr[i]) != 0:
                print('{0} data error'.format(i))
                break
        print(dgtr.shape)
        print(np.sum(self.tsr[:,-1]))
        np.save('data/groundTruths_density.npy', dgtr)
        self.gtr = self.gtr[:,:-1]
        self.tsr = self.tsr[:,:-1]
        print('after split')
        print('raw trainingSets', self.tsr.shape)
        print('raw groundTruth: ', self.gtr.shape)
        print('split complete\n')

    # only save the first sample after 30 seconds
    def from30toEnd(self):
        # self.gtr = self.gtr[:1, 30:]
        # self.tsr = self.tsr[:1, 30:]
        self.gtr = self.gtr[:, 30:]
        self.tsr = self.tsr[:, 30:]
        print(self.tsr.shape)
        print(self.gtr.shape)
        print('from30toEnd complete\n')

    # switch all elements to zero or one 
    def oneOrZero(self):
        m = np.median(self.gtr[self.gtr!=0])
        print('median:',m)
        # self.gtr[self.gtr<=m] = 0
        # self.gtr[self.gtr>m] = 1
        self.gtr[self.gtr<m] = 0
        self.gtr[self.gtr>=m] = 1
        print('oneOrZero complete\n')


    # ground truth only save the last second (the 30th second)
    def lastSecond(self):
        gtr1 = self.gtr[:,29:,:,:].reshape((1, 16, 16))
        print('self.gtr[:,29:,:,:]: ', self.gtr[:,29:,:,:].shape)
        print('gtr1: ', gtr1.shape)
        print('self.gtr == gtr1:', np.all(gtr1==self.gtr[:,29]))
        self.gtr = gtr1
        print('lastSecond complete\n')

    # print number of non-zeros and zeros
    def computeWeights(self):
        one = self.gtr[self.gtr>0].size
        zero = self.gtr[self.gtr==0].size
        print('zero:',zero)
        print('one:',one)
        print('weight:',zero/one)
        print('computeWeights complete\n')


    # nomalize groud truth as the last second
    def batchNormalize(self):
        self.gtr = (self.gtr - np.min(self.gtr)) / (np.max(self.gtr) - np.min(self.gtr))
        print('min: ', np.min(self.gtr))
        print('max: ', np.max(self.gtr))
        print('mean: ', np.mean(self.gtr))
        print('median: ', np.median(self.gtr))
        print('batchNormalize complete\n')

    # broadcast one sample to many 
    def broadCast(self):
        self.tsr = np.broadcast_to(self.tsr, (10000, 30, 16, 16, 4))
        self.gtr = np.broadcast_to(self.gtr, (10000, 30, 16, 16))
        print(self.tsr.shape)
        print(self.gtr.shape)
        print('broadCast complete\n')
        

    def saveData(self):
        np.save('data/trainingSets_diff.npy', self.tsr)
        np.save('data/groundTruths_diff.npy', self.gtr)
        print('save complete\n')

    def checkGroundTruthIdentical(self):
        a1 = self.gtr[0]
        a2 = self.gtr[10]
        a3 = self.gtr[100]
        a4 = self.gtr[1000]

        print(np.all(a1==a2))
        print(np.all(a1==a3))
        print(np.all(a1==a4))
        print(np.all(a3==a2))
        print('check complete\n')

p = Preprocess()
# p.split()
p.from30toEnd()
p.oneOrZero()
p.computeWeights()
# p.broadCast()
p.checkGroundTruthIdentical()
p.saveData()
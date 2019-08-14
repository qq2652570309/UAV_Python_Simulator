import numpy as np


class Preprocess:

    def __init__(self):
        self.gtr = np.load("data/groundTruths_raw.npy")
        self.tsr = np.load("data/trainingSets_raw.npy")
        print('raw trainingSets', self.tsr.shape)
        print('raw groundTruth: ', self.gtr.shape)

    def splitByTime(self, start=0, end=0):
        if end == 0:
            self.gtr = self.gtr[:, start:]
            self.tsr = self.tsr[:, start:]
        else:
            self.gtr = self.gtr[:, start:end]
            self.tsr = self.tsr[:, start:end]
        print(self.tsr.shape)
        print(self.gtr.shape)
        print('splitByTime complete\n')

    # only save the first sample after 30 seconds
    def from30toEnd(self):
        # self.gtr = self.gtr[:1, 30:]
        # self.tsr = self.tsr[:1, 30:]
        self.gtr = self.gtr[:, 20:]
        self.tsr = self.tsr[:, 20:]
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
        for i in range(len(self.gtr)):
            self.gtr[i] = (self.gtr[i] - np.min(self.gtr[i])) / (np.max(self.gtr[i]) - np.min(self.gtr[i]))
        print('min: ', np.min(self.gtr))
        print('max: ', np.max(self.gtr))
        print('mean: ', np.mean(self.gtr))
        print('median: ', np.median(self.gtr))
        print('batchNormalize complete\n')

    # broadcast one sample to many 
    def broadCast(self):
        self.tsr = np.broadcast_to(self.tsr, (10000, 30, 32, 32, 4))
        self.gtr = np.broadcast_to(self.gtr, (10000, 30, 32, 32))
        print(self.tsr.shape)
        print(self.gtr.shape)
        print('broadCast complete\n')
        
    # (30, 32, 32) --> (32, 32)
    def generateDensity(self):
        self.gtr = np.sum(self.gtr, axis=1)
        print(self.gtr.shape)
        print('generateDensity complete\n')

    def saveData(self):
        np.save('data/trainingSets_diff.npy', self.tsr)
        np.save('data/groundTruths_diff.npy', self.gtr)
        print('trainingSets shape', self.tsr.shape)
        print('groundTruths shape', self.gtr.shape)
        print('save complete\n')

    def checkGroundTruthIdentical(self):
        p = np.random.randint(0, len(self.gtr), 5)
        for i in range(1,5):
            print(np.all(self.gtr[p[i]] == self.gtr[p[i-1]]))
        print('check complete\n')

p = Preprocess()
p.splitByTime(20)
# p.from30toEnd()
p.oneOrZero()
p.generateDensity()
p.batchNormalize()
p.computeWeights()
# p.broadCast()
p.checkGroundTruthIdentical()
p.saveData()
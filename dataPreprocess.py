import numpy as np
from simulator import Simulator
import logging
import time

class Preprocess:

    def __init__(self, groundTruth=None, trainingSets=None):
        if groundTruth is None:
            self.gtr = np.load("data/row_groundTruths.npy")
        # elif '.npy' in groundTruth:
        #     self.gtr = np.load(groundTruth)
        else:
            self.gtr = groundTruth

        if trainingSets is None:
            self.tsr = np.load("data/row_trainingSets.npy")
        # elif '.npy' in trainingSets:
        #     self.tsr = np.load(trainingSets)
        else:
            self.tsr = trainingSets
        logging.info('---Initial Shape---')
        print('---Initial Shape---')
        print('raw trainingSets', self.tsr.shape)
        print('raw groundTruth: ', self.gtr.shape)

    def splitByTime(self, start=0, end=0):
        if end == 0:
            self.gtr = self.gtr[:, start:]
            self.tsr = self.tsr[:, start:]
        else:
            self.gtr = self.gtr[:, start:end]
            self.tsr = self.tsr[:, start:end]
        logging.info(self.tsr.shape)
        logging.info(self.gtr.shape)
        logging.info('splitByTime complete\n')

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
        logging.info('oneOrZero complete\n')


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
        logging.info('zero: {0}'.format(zero))
        logging.info('one: {0}'.format(one))
        logging.info('weight: {0}'.format(zero/one))
        logging.info('computeWeights complete\n')

    # nomalize groud truth as the last second
    def batchNormalize(self):
        for i in range(len(self.gtr)):
            self.gtr[i] = (self.gtr[i] - np.min(self.gtr[i])) / (np.max(self.gtr[i]) - np.min(self.gtr[i]))
        logging.info('min: {0}'.format(np.min(self.gtr)))
        logging.info('max: {0}'.format(np.max(self.gtr)))
        logging.info('mean: {0}'.format(np.mean(self.gtr)))
        logging.info('median: {0}'.format(np.median(self.gtr)))
        logging.info('batchNormalize complete\n')

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
        logging.info(self.gtr.shape)
        logging.info('generateDensity complete\n')

    def saveData(self, name='density'):
        np.save('data/trainingSets_{0}.npy'.format(name), self.tsr)
        np.save('data/groundTruths_{0}.npy'.format(name), self.gtr)
        print('trainingSets shape', self.tsr.shape)
        print('groundTruths shape', self.gtr.shape)
        print('save complete\n')

    def checkGroundTruthIdentical(self):
        p = np.random.randint(0, len(self.gtr), 5)
        for i in range(1,5):
            logging.info(np.all(self.gtr[p[i]] == self.gtr[p[i-1]]))
        logging.info('check complete\n')

    def averageLaunchingNumber(self):
        sum1 = np.sum(self.tsr[:,:, 0:4, 0:4, 0])
        sum2 = np.sum(self.tsr[:,:, 22:26, 23:26, 0])
        sum3 = np.sum(self.tsr[:,:, 27:31, 27:31, 0])
        sampleNum = self.tsr.shape[0]
        timeTotal = self.tsr.shape[1]
        ave1 = sum1 / sampleNum / timeTotal * 5
        ave2 = sum2 / sampleNum / timeTotal * 5
        ave3 = sum3 / sampleNum / timeTotal * 5
        print('In area1, average number of UAV launched: ', ave1)
        print('In area2, average number of UAV launched: ', ave2)
        print('In area3, average number of UAV launched: ', ave3)
        print('average lauching complete\n')


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.disabled = False
    logging.basicConfig(filename='log.txt', format='%(levelname)s:%(message)s', level=logging.INFO)
    logging.info('Started')

    s = Simulator(iteration=10, row=32, column=32, time=90)
    startTimeTotal = time.time()
    s.generate()
    
    print('avg flying time: ', s.totalFlyingTime/s.totalUavNum)

    logging.info('Finished')
    np.save('data/row_trainingSets.npy', s.trainingSets)
    np.save('data/row_groundTruths.npy', s.groundTruths)
    np.save('data/positions.npy', s.positions)
    print('avg flying time: ', s.totalFlyingTime/s.totalUavNum)
    # logging.info('finish generate, cost {0} \n'.format(time.time() - startTimeTotal))
    logging.info('avg flying time: {0} \n'.format( s.totalFlyingTime/s.totalUavNum))

    p_traj = Preprocess()
    p_traj.splitByTime(30)
    p_traj.oneOrZero()
    p_traj.computeWeights()
    p_traj.checkGroundTruthIdentical()
    p_traj.saveData('trajectory')

    p_den = Preprocess()
    p_den.splitByTime(30)
    p_den.oneOrZero()
    p_den.generateDensity()
    p_den.batchNormalize()
    p_den.computeWeights()
    p_den.checkGroundTruthIdentical()
    p_den.saveData('density')

    logging.info('Finished dataPreprocess')
    print('Finished dataPreprocess')

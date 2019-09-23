import numpy as np
from simulator import Simulator
import logging
import time
import os

class Preprocess:

    def __init__(self, groundTruth=None, trainingSets=None):
        if groundTruth is None:
            print("ground truth is none")
        else:
            self.gtr = groundTruth
        if trainingSets is None:
            print("training set is none")
        else:
            self.tsr = trainingSets
        
        logging.info('---Initial Shape---')
        print('---Initial Shape---')
        print('raw trainingSets', self.tsr.shape)
        print('raw groundTruth: ', self.gtr.shape)

    # save data from start to end
    def splitByTime(self, start=0, end=0):
        if end == 0:
            self.gtr = self.gtr[:, start:]
            # self.tsr = self.tsr[:, start:]
        else:
            self.gtr = self.gtr[:, start:end]
            # self.tsr = self.tsr[:, start:end]
        logging.info(self.tsr.shape)
        logging.info(self.gtr.shape)
        logging.info('splitByTime complete\n')


    # switch all elements to zero or one 
    def oneOrZero(self, gtr):
        m = np.median(gtr[gtr!=0])
        logging.info('median: {0}'.format(m))
        # self.gtr[self.gtr<=m] = 0
        # self.gtr[self.gtr>m] = 1
        gtr[gtr<m] = 0
        gtr[gtr>=m] = 1
        logging.info('oneOrZero complete\n')
        return gtr


    def densityToOne(self, gtr):
        gtr[gtr>0] = 1
        logging.info('densityToOne complete\n')
        return gtr


    # ground truth only save the last second (the 30th second)
    def lastSecond(self):
        gtr1 = self.gtr[:,29:,:,:].reshape((1, 16, 16))
        print('self.gtr[:,29:,:,:]: ', self.gtr[:,29:,:,:].shape)
        print('gtr1: ', gtr1.shape)
        print('self.gtr == gtr1:', np.all(gtr1==self.gtr[:,29]))
        self.gtr = gtr1
        print('lastSecond complete\n')

    # print number of non-zeros and zeros
    def computeWeights(self, gtr):
        one = gtr[gtr>0].size
        zero = gtr[gtr==0].size
        logging.info('zero: {0}'.format(zero))
        logging.info('one: {0}'.format(one))
        logging.info('weight: {0}'.format(zero/one))
        logging.info('computeWeights complete\n')

    # nomalize groud truth as the last second
    def batchNormalize(self, gtr):
        for i in range(len(gtr)):
            gtr[i] = (gtr[i] - np.min(gtr[i])) / (np.max(gtr[i]) - np.min(gtr[i]))
        logging.info('min: {0}'.format(np.min(gtr)))
        logging.info('max: {0}'.format(np.max(gtr)))
        logging.info('mean: {0}'.format(np.mean(gtr)))
        logging.info('median: {0}'.format(np.median(gtr)))
        logging.info('batchNormalize complete\n')
        return gtr


    # lumped map divided time, return with batch normalization
    def averageDensity(self, gtr, time):
        gtr = gtr/time
        return self.batchNormalize(gtr)


    # broadcast one sample to many 
    def broadCast(self):
        self.tsr = np.broadcast_to(self.tsr, (10000, 30, 32, 32, 4))
        self.gtr = np.broadcast_to(self.gtr, (10000, 30, 32, 32))
        print(self.tsr.shape)
        print(self.gtr.shape)
        print('broadCast complete\n')
        
    # (30, 32, 32) --> (32, 32)
    def generateDensity(self, gtr):
        temp = np.sum(gtr, axis=1)
        logging.info(gtr.shape)
        logging.info('generateDensity complete\n')
        return temp    

    def save(self, data, name='feature', directory='test'):
        # if not os.path.exists('../../../data/zzhao/uav_regression/{0}'.format(directory)):
        #     os.mkdir('../../../data/zzhao/uav_regression/{0}'.format(directory))
        #     os.chmod('../../../data/zzhao/uav_regression/{0}'.format(directory), 0o777)
        if name is 'feature':
            print('training_data_trajectory shape is {0}'.format(data.shape))
            np.save('data/training_data_trajectory.npy'.format(directory), data)
            os.chmod('data/training_data_trajectory.npy'.format(directory), 0o777)
        elif name is 'cnn':
            print('training_label_density shape is {0}'.format(name))
            np.save('../../../data/zzhao/uav_regression/{0}/training_label_density.npy'.format(directory), data)
            os.chmod('../../../data/zzhao/uav_regression/{0}/training_label_density.npy'.format(directory), 0o777)
        elif name is 'lstm':
            print('training_label_trajectory.npy shape is {0}'.format(name))
            np.save('../../../data/zzhao/uav_regression/{0}/training_label_density.npy'.format(directory), data)
            os.chmod('../../../data/zzhao/uav_regression/{0}/training_label_density.npy'.format(directory), 0o777)
        elif name is 'pattern':
            print('training_label_trajectory.npy shape is {0}'.format(name))
            np.save('data/training_label_density.npy'.format(directory), data)
            os.chmod('data/training_label_density.npy'.format(directory), 0o777)
        else:
            print('stop')
        print('{0} save complete\n'.format(name))

    def checkGroundTruthIdentical(self, gtr):
        p = np.random.randint(0, len(gtr), 5)
        for i in range(1,5):
            logging.info(np.all(gtr[p[i]] == gtr[p[i-1]]))
        logging.info('check complete\n')
    
    def checkDataIdentical(self, data1, data2):
        # p = np.random.randint(0, len(data1), 5)
        for i in range(0,5):
            logging.info(np.all(data1[i] == data2[i]))
        logging.info('check complete\n')


    def compressTime(self):
        # feature: (10, 240, 100, 100, 2)
        # label  : (10, 240, 100, 100)
        # nf : (10,24,100,100,2)
        # nl : (10,24,100,100)
        nf = np.zeros((self.tsr.shape[0],int(self.tsr.shape[1]/10),self.tsr.shape[2],self.tsr.shape[3],self.tsr.shape[4]))
        nl = np.zeros((self.gtr.shape[0],int(self.gtr.shape[1]/10),self.gtr.shape[2],self.gtr.shape[3]))

        for i in range(10):
            ft, lb = self.tsr[i], self.gtr[i]
            for it in range(10, 241, 10):
                time_idx = int(it/10)-1
                nl[i, time_idx] = np.sum(lb[it-10:it], axis=0)/10
                nf[i,time_idx,:,:,1] = lb[it-1,:,:]/10
                nf[i,time_idx,:,:,1] = (nf[i,time_idx,:,:,1] - np.min(nf[i,time_idx,:,:,1])) / (np.max(nf[i,time_idx,:,:,1]) - np.min(nf[i,time_idx,:,:,1]))
                nf[i,time_idx,:,:,0] = ft[it-1,:,:,0]
            nl[i] = self.batchNormalize(nl[i])
        self.tsr = nf
        self.gtr = nl


    def featureLabel(self, directory='test'):
        logging.info('generating lstm feature\n')
        self.save(self.tsr, 'feature', directory=directory)
        
        logging.info('generating pattern labels\n')
        patternGt = np.copy(self.gtr)
        # patternGt = self.oneOrZero(patternGt)
        # patternGt = self.generateDensity(patternGt)
        # patternGt = self.averageDensity(patternGt, 60)
        self.save(patternGt, 'pattern', directory=directory)
        
        print('finish saving')



if __name__ == "__main__":
    logger = logging.getLogger()
    logger.disabled = False
    logging.basicConfig(filename='log.txt', format='%(levelname)s:%(message)s', level=logging.INFO)
    logging.info('Started')

    
    s = Simulator(batch=1000, row=100, column=100, time=240)
    startTimeTotal = time.time()
    s.generate()
    
    print('avg flying time: ', s.totalFlyingTime/s.totalUavNum)

    print(s.trainingSets.shape)
    print(s.groundTruths.shape)
    
    # feature: (10, 200, 100, 100, 1)
    # label:   (10, 200, 100, 100)

    p = Preprocess(trainingSets=s.trainingSets, groundTruth=s.groundTruths)
    p.compressTime()
    p.featureLabel(directory='cnn')
    
    
    '''
    logging.info('Finished')
    print('avg flying time: ', s.totalFlyingTime/s.totalUavNum)
    # logging.info('finish generate, cost {0} \n'.format(time.time() - startTimeTotal))
    logging.info('avg flying time: {0} \n'.format( s.totalFlyingTime/s.totalUavNum))
    
    p = Preprocess(trainingSets=s.trainingSets, groundTruth=s.groundTruths)
    # p.splitByTime(30)
    p.featureLabel()

    logging.info('Finished dataPreprocess')
    print('Finished dataPreprocess')
    '''

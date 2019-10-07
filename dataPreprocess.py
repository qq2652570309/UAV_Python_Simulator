import numpy as np
import logging
import time
import os

from simulator import Simulator
from simulator1 import Simulator1
from simulator2 import Simulator2
from simulator3 import Simulator3
from simulator4 import Simulator4

class Preprocess:

    def __init__(self, label=None, pfeature=None, rfeature=None):
        if label is None:
            print("ground truth is none")
        else:
            self.gtr = label

        if pfeature is None:
            print("pnet featuret is none")
        else:
            self.pfeature = pfeature

        if rfeature is None:
            print("rnet feature is none")
        else:
            self.rfeature = rfeature
        
        logging.info('')
        logging.info('---Initial Shape---')
        logging.info('  initial pnet feature: {0}'.format(self.pfeature.shape))
        logging.info('  initial rnet feature: {0}'.format(self.rfeature.shape))
        logging.info('  initial label: {0}\n'.format(self.gtr.shape))
        print('---Initial Shape---')
        print('initial pnet feature', self.pfeature.shape)
        print('initial rnet feature', self.rfeature.shape)
        print('initial label: ', self.gtr.shape)
        print('')

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


    def checkDataIdentical(self, data1, data2):
        # p = np.random.randint(0, len(data1), 5)
        for i in range(0,5):
            logging.info(np.all(data1[i] == data2[i]))
        logging.info('check complete\n')

    def compressTime(self):
        # feature: (n, 200, 10,  5) --> (n, 20, 100, 5)
        # label  : (n, 200, 100, 100) --> (n, 20, 100, 100)
        # nf : (n, 20, 100, 5)
        # nl : (n, 20, 100, 100)

        nf = np.zeros((self.tsr.shape[0],int(self.tsr.shape[1]/10),self.tsr.shape[2]*10,self.tsr.shape[3]))
        nl = np.zeros((self.gtr.shape[0],int(self.gtr.shape[1]/10),self.gtr.shape[2],self.gtr.shape[3]))

        for i in range(10):
            ft, lb = self.tsr[i], self.gtr[i]
            for it in range(10, 201, 10):
                time_idx = int(it/10)-1
                # every sample, generate density map in 10 time intervel
                nl[i, time_idx] = np.sum(lb[it-10:it], axis=0)/10
                # every sample, record all task
                task_num = 0
                for j in range(it-10, it):
                    for k in range(10):
                        nf[i, time_idx, task_num, :] = ft[i, j, k, :]
                        task_num+=1
            nl[i] = self.batchNormalize(nl[i])
        self.tsr = nf
        self.gtr = nl

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
        logging.info('      after batchNormalize')
        logging.info('          min: {0}'.format(np.min(gtr)))
        logging.info('          max: {0}'.format(np.max(gtr)))
        logging.info('          mean: {0}'.format(np.mean(gtr)))
        logging.info('          median: {0}'.format(np.median(gtr)))
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
        logging.info('      density map is {0}'.format(temp.shape))
        return temp    


    def generatePattern(self, gtr):
        temp = np.sum(gtr, axis=1)
        temp[temp>0] = 1
        logging.info(temp.shape)
        logging.info('generatePattern complete\n')
        return temp


    def save(self, data, name='test', direcoty='test'):
        if not os.path.exists('../../../data/zzhao/uav_regression/{0}'.format(direcoty)):
            os.mkdir('../../../data/zzhao/uav_regression/{0}'.format(direcoty))
            os.chmod('../../../data/zzhao/uav_regression/{0}'.format(direcoty), 0o777)
        if 'lkjsdf' in name:
            return
        if 'data' in name:
            if 'density' in name:
                print(' {0} is {1}'.format(name, data.shape))
                np.save('../../../data/zzhao/uav_regression/{0}/{1}.npy'.format(direcoty, name), data)
                os.chmod('../../../data/zzhao/uav_regression/{0}/{1}.npy'.format(direcoty, name), 0o777)
            elif 'task' in name:
                print(' {0} is {1}'.format(name, data.shape))
                np.save('../../../data/zzhao/uav_regression/{0}/{1}.npy'.format(direcoty, name), data)
                os.chmod('../../../data/zzhao/uav_regression/{0}/{1}.npy'.format(direcoty, name), 0o777)
            elif 'probability' in name:
                print(' {0} is {1}'.format(name, data.shape))
                np.save('../../../data/zzhao/uav_regression/{0}/{1}.npy'.format(direcoty, name), data)
                os.chmod('../../../data/zzhao/uav_regression/{0}/{1}.npy'.format(direcoty, name), 0o777)
            else:
                print('No such feature! \n')
        elif 'label' in name:
            print(' {0} is {1}'.format(name, data.shape))
            np.save('../../../data/zzhao/uav_regression/{0}/{1}.npy'.format(direcoty, name), data)
            os.chmod('../../../data/zzhao/uav_regression/{0}/{1}.npy'.format(direcoty, name), 0o777)
        else:
            print('No such data! \n')
        print(' {0} save complete\n'.format(name))


    # generate density map from timestep start -> end
    def intervalDensity(self, data, start, end):
        logging.info('      generate density map from {0} to {1}'.format(start, end))
        interval = data[:,start:end]
        densityMap = self.generateDensity(interval)
        return self.averageDensity(densityMap, end-start)


    def featureLabel(self, direcoty='test'):        
        logging.info('  process labels:')
        trajectory = np.copy(self.gtr)
        # densityLabel = (batch, 100, 100) in last 10 timesteps
        densityLabel = self.intervalDensity(trajectory, trajectory.shape[1]-10, trajectory.shape[1])
        self.save(densityLabel, name='label_density', direcoty=direcoty)

        logging.info('')
        logging.info('  process Pnet feature:')
        # desntiyFeature = (batch, 100, 100) in first 10 timesteps
        desntiyFeature = self.intervalDensity(trajectory, 0, 10)
        self.save(desntiyFeature, name='data_density', direcoty=direcoty)
        # feature_task = (batch, 60, 15, 5)
        self.save(self.pfeature, name='data_tasks', direcoty=direcoty)
        logging.info('')
        logging.info('  process Rnet feature:')
        self.save(self.rfeature, name='data_probability_pattern', direcoty=direcoty)
        print('finish saving')


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.disabled = False
    logging.basicConfig(filename='log.txt', format='%(levelname)s:%(message)s', level=logging.INFO)
    logging.info('Started')


    # s = Simulator1(batch=1, time=200, mapSize=100, taskNum=15, trajectoryTime=110, taskTime=50)
    # s = Simulator2(batch=1, time=200, mapSize=100, taskNum=15)
    s = Simulator3(batch=1, time=200, mapSize=100, taskNum=15)
    # s = Simulator4(batch=1, time=200, mapSize=100, taskNum=15)
    startTimeTotal = time.time()
    s.generate()
    logging.info('Simulater Finished')
    logging.info('  generation costs {0} \n'.format(time.time() - startTimeTotal))
    print('avg flying time: ', s.totalFlyingTime/s.totalUavNum)
    logging.info('avg flying time: {0} \n'.format( s.totalFlyingTime/s.totalUavNum))
    print('total tasks number: ', s.totalUavNum)
    logging.info('total tasks number: {0} \n'.format(s.totalUavNum))

    p = Preprocess(pfeature=s.tasks, label=s.trajectors, rfeature=s.Rfeature)
    # p.featureLabel(direcoty='test')

    logging.info('Finished dataPreprocess')
    print('Finished dataPreprocess')

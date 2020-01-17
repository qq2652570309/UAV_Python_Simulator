import numpy as np
# from simulator import Simulator
# from simulatorNFZ import SimulatorNFZ
<<<<<<< HEAD
# from simulator_astar import SimulatorAstar
from simulator_astarProb import SimulatorAstarProb
=======
from simulator_astar import SimulatorAstar
>>>>>>> bafe927d9ca7472df95e2243eb12e84659ff7aaa
import logging
import time
import os

np.random.seed(0)

# from simulator import Simulator
# from simulator_randTask import Simulator
# from simulator_routing import Simulator


class Preprocess:

    def __init__(self, label=None, mainList=None, subOutput=None, subOutputCube=None, rfeature=None, subList=None, subLabel=None, noFlyZone=None):
        logging.info('')
        logging.info('---Initial Shape---')
        print('\n---Initial Shape---')
        if label is None:
            print("MainNet label is none")
        else:
            self.gtr = label
            logging.info('  initial MainNet label: {0}\n'.format(self.gtr.shape))
            print('initial MainNet label: ', self.gtr.shape)

        if mainList is None:
            print("MainNet tasklist is none")
        else:
            self.mt = mainList
            logging.info('  initial MainNet tasklist: {0}'.format(self.mt.shape))
            print('initial MainNet tasklist', self.mt.shape)

        if subOutput is None:
            print("subNet Output is none")
        else:
            self.subOutput = subOutput
            logging.info('  initial subNet Output: {0}'.format(self.subOutput.shape))
            print('initial subNet Output', self.subOutput.shape)
        
        if subOutputCube is None:
            print("subNet Output Cube is none")
        else:
            self.subOutputCube = subOutputCube
            logging.info('  initial subNet Cube Output: {0}'.format(self.subOutputCube.shape))
            print('initial subNet Cube Output', self.subOutputCube.shape)

        if rfeature is None:
            print("rnet feature is none")
        else:
            self.rfeature = rfeature
            logging.info('  initial rnet feature: {0}'.format(self.rfeature.shape))
            print('initial rnet feature', self.rfeature.shape)

        if subList is None:
            print("subNet tasklist is none")
        else:
            self.st = subList
            logging.info('  initial subNet tasklist: {0}\n'.format(self.st.shape))
            print('initial subNet tasklist: {0}'.format(self.st.shape))

        if subLabel is None:
            print("subLabel is none")
        else:
            self.sl = subLabel
            logging.info('  initial subLabel: {0}\n'.format(self.sl.shape))
            print('initial subLabel: {0}\n'.format(self.sl.shape))
        
        if noFlyZone is None:
            print("noFlyZone is none")
        else:
            self.nfz = noFlyZone
            logging.info('  noFlyZone: {0}\n'.format(self.nfz.shape))
            print('noFlyZone: {0}\n'.format(self.nfz.shape))

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

    def standardDeviation(self, interval=15):
        # lb: (n, 60, 100, 100) --> (n, 100, 100)
        # nf: (n, 60, 100, 100) --> (n, 30, 100, 100)
        batchNum, intervalNum, row, col = self.gtr.shape
        intervalNum -= 2*interval
        lb = np.zeros((batchNum, row, col))
        nf = np.zeros((batchNum, intervalNum, row, col))
        for b in range(batchNum):
            lb[b] = np.sum(self.gtr[b, intervalNum+interval:intervalNum+interval*2], axis=0) 
            for i in range(intervalNum):
                print(i, i+interval)
                nf[b, i] = np.sum(self.gtr[b, i:i+interval], axis=0)

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
        # for i in range(len(gtr)):
        #     gtr[i] = (gtr[i] - np.min(gtr[i])) / (np.max(gtr[i]) - np.min(gtr[i]))
        if np.max(gtr) != 0:
            gtr = (gtr - np.min(gtr)) / (np.max(gtr) - np.min(gtr))
        # logging.info('      after batchNormalize')
        # print('          min: {0}'.format(np.min(gtr)))
        # print('          max: {0}'.format(np.max(gtr)))
        # print('          mean: {0}'.format(np.mean(gtr)))
        # print('          median: {0}'.format(np.median(gtr)))
        return gtr

    # lumped map divided time, return with batch normalization
    def averageDensity(self, gtr, time):
        gtr = gtr/time
        return gtr

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

    def save(self, data, name='test', directory='test', subDirectory='subtest'):
        if not os.path.exists('../../../data/zzhao/uav_regression/{0}'.format(directory)):
            os.mkdir('../../../data/zzhao/uav_regression/{0}'.format(directory))
            os.chmod('../../../data/zzhao/uav_regression/{0}'.format(directory), 0o777)
        if not os.path.exists('../../../data/zzhao/uav_regression/{0}/{1}'.format(directory, subDirectory)):
            os.mkdir('../../../data/zzhao/uav_regression/{0}/{1}'.format(directory, subDirectory))
            os.chmod('../../../data/zzhao/uav_regression/{0}/{1}'.format(directory, subDirectory), 0o777)
        
        if os.path.exists('../../../data/zzhao/uav_regression/{0}/{1}/{2}.npy'.format(directory, subDirectory, name)):
            os.remove('../../../data/zzhao/uav_regression/{0}/{1}/{2}.npy'.format(directory, subDirectory, name))

        np.save('../../../data/zzhao/uav_regression/{0}/{1}/{2}.npy'.format(directory, subDirectory, name), data)
        os.chmod('../../../data/zzhao/uav_regression/{0}/{1}/{2}.npy'.format(directory, subDirectory, name), 0o777)
        print(' {0}/{1}: {2} save complete\n'.format(subDirectory, name, data.shape))


    # generate density map from timestep start -> end
    def intervalDensity(self, data, start, end):
        logging.info('      generate density map from {0} to {1}'.format(start, end))
        interval = data[:,start:end]
        densityMap = self.generateDensity(interval)
        return self.averageDensity(densityMap, end-start)
    
    def featureLabel(self, directory='test'):
        # ---------------------- main network ----------------------      
        logging.info('  process labels:')
        trajectory = np.copy(self.gtr)
        # densityLabel = (batch, 100, 100) in last 10 timesteps
        densityLabel = self.intervalDensity(trajectory, trajectory.shape[1]-10, trajectory.shape[1])
        self.save(densityLabel, name='label_mainnet', directory=directory, subDirectory='fushion')

        logging.info('')
        logging.info('  process features:')
        # desntiyFeature = (batch, 100, 100) in first 10 timesteps
        desntiyFeature = self.intervalDensity(trajectory, 0, 10)
        self.save(desntiyFeature, name='data_init', directory=directory, subDirectory='fushion')
        # tasklist = (batch, 60, 15, 5)
        self.save(self.mt, name='data_tasks', directory=directory, subDirectory='fushion')
        # subnet output = (batch, 60, 100, 100)
        self.save(self.subOutput, name='data_subnet', directory=directory, subDirectory='fushion')
        logging.info('')
        # subnet Cube output = (batch, 60, 100, 100)
<<<<<<< HEAD
        # self.save(self.subOutputCube, name='data_subnet_cube', directory=directory, subDirectory='fushion')
        # logging.info('')
=======
        self.save(self.subOutputCube, name='data_subnet_cube', directory=directory, subDirectory='fushion')
        logging.info('')
>>>>>>> bafe927d9ca7472df95e2243eb12e84659ff7aaa
        
        # ---------------------- sub network ----------------------
        logging.info('  process subnet label:')
        self.save(self.sl, name="label_subnet", directory=directory, subDirectory='subnet')
        logging.info('  process subnet tasklist:')
        self.save(self.st, name="data_tasklist", directory=directory, subDirectory='subnet')
        # ---------------------- No Fly Zone ----------------------
        logging.info('  process subnet label:')
        self.save(self.nfz, name="data_nfz", directory=directory, subDirectory='fushion')

        print('finish saving')


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.disabled = False
    logging.basicConfig(filename='log.txt', format='%(levelname)s:%(message)s', level=logging.INFO)
    logging.info('Started')

<<<<<<< HEAD
    s = SimulatorAstarProb(batch=3000, time=200, mapSize=100, taskNum=15, trajectoryTime=70, taskTime=60)
=======
    s = SimulatorAstar(batch=3000, time=200, mapSize=100, taskNum=15, trajectoryTime=70, taskTime=60)
>>>>>>> bafe927d9ca7472df95e2243eb12e84659ff7aaa
    startTimeTotal = time.time()
    s.generate()
    
    logging.info('Simulater Finished')
    logging.info('  generation costs {0} \n'.format(time.time() - startTimeTotal))
    print('avg flying time: ', s.totalFlyingTime/s.totalUavNum)
    logging.info('avg flying time: {0} \n'.format( s.totalFlyingTime/s.totalUavNum))
    print('total tasks number: ', s.totalUavNum)
    logging.info('total tasks number: {0} \n'.format(s.totalUavNum))

    p = Preprocess(mainList=s.mainTaskList, label=s.trajectors,
                    subOutput=s.subOutput, subOutputCube=s.subOutputCube, 
                    rfeature=s.Rfeature, noFlyZone=s.NFZ,
                    subLabel=s.subLabel, subList=s.subTaskList)
<<<<<<< HEAD
    p.featureLabel(directory='astarProbility')
=======
    p.featureLabel(directory='astar')
>>>>>>> bafe927d9ca7472df95e2243eb12e84659ff7aaa
    
    logging.info('Finished dataPreprocess')
    print('Finished dataPreprocess')


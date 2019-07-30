import time
import logging
import numpy as np
from simulator import Simulator

logger = logging.getLogger()
logger.disabled = True

logging.basicConfig(filename='log.txt', format='%(levelname)s:%(message)s', level=logging.INFO)

logging.info('Started')
startTimeIter = time.time()
# s = Simulator(iteration=2, row=4, column=4, time=5, startPointsNum=3, endPointsNum=3)
s = Simulator(iteration=10000, row=16, column=16, time=61, startPointsNum=15, endPointsNum=15)
# s = Simulator(iteration=100, row=32, column=32, time=32, startPointsNum=100, endPointsNum=100)
s.generate()
# s.dataProcess()
logging.info('Finished')
np.save('data/trainingSets_raw.npy', s.trainingSets)
np.save('data/groundTruths_raw.npy', s.groundTruths)
print('total time: ', time.time() - startTimeIter)
# logging.info('trainingSets: \n{0}'.format(s.trainingSets))
# logging.info('groundTruths: \n{0}'.format(s.groundTruths))

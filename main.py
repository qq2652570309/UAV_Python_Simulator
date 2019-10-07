from model import Lstm_Cnn_Model
from dataPreprocess import Preprocess
from simulator import Simulator
import numpy as np
from generateImage import Image

'''
s = Simulator(batch=10, row=100, column=100, time=240)
s.generate()

p = Preprocess(trainingSets=s.trainingSets, groundTruth=s.groundTruths)
p.compressTime()
'''


CSM = Lstm_Cnn_Model(
        epics=10,
        weight=6.44
    )

CSM.lstmLayers()

'''
x = p.tsr
y = p.gtr
'''
x = np.load('data/training_data_trajectory.npy')
y = np.load('data/training_label_density.npy')
x = x[:10]
y = y[:10]

print(x.shape)
print(y.shape)
CSM.model.load_weights('checkpoints/mse-01-0.00.hdf5')
CSM.configure('mse')
result = CSM.model.evaluate(x, y, batch_size=1)
print('result: ', result)
prediction = CSM.model.predict(x)
print(prediction.shape)
# prediction = np.round(np.clip(prediction, 0, 1))
# np.save('data/evaluate_cnn.npy', prediction)
# np.save('data/y_test.npy', y)





a = prediction[0,10]
median = np.median(a)
a[a>=median] = 1
a[a<median] = 0


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.gray()
plt.imshow(a)
plt.savefig('img/test.png')

'''
data = [
        y,
        prediction,
    ]
rowHeader = ['groundTrue', 'prediction']

i = Image(data, rowHeader, 'test')
i.generate()
'''
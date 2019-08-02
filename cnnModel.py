import os
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Dropout, LSTM
from tensorflow.keras.layers import Flatten, Activation, Reshape
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, TimeDistributed
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras import losses, metrics
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Cnn_Model:
    def __init__(self, trs=None, grt=None, epics=1):
        self.model = None
        self.epics = epics

        # uav_data = np.load(trs)
        # print('uav_data: ', uav_data.shape) # (10000, 30, 32, 32, 4)

        # uav_label = np.load(grt)
        # print('uav_label: ', uav_label.shape) # (10000, 30, 32, 32)

        # data_size = int(len(uav_data) * 0.85)

        # (x_train, y_train) = uav_data[:data_size], uav_label[:data_size]
        # (x_test, y_test) = uav_data[data_size:], uav_label[data_size:]

        cnn_model = Sequential()
        cnn_model.add(Conv1D(8, kernel_size=2,
                        activation='relu',
                        input_shape=(32, 32)))
        cnn_model.add(Conv1D(16, kernel_size=2, activation='relu'))
        cnn_model.add(MaxPooling1D(pool_size=2))
        # cnn_model.add(Conv1D(32, kernel_size=(2, 2), activation='relu'))
        # cnn_model.add(Conv1D(64, kernel_size=(3, 3), activation='relu'))
        # cnn_model.add(MaxPooling2D(pool_size=(2,2)))
        cnn_model.add(Flatten())
        cnn_model.summary()
        cnn_model.add(Dense(240))
        cnn_model.add(Reshape((15, 15)))
        # cnn_model.add(Conv1DTranspose(32, kernel_size=(5, 5), activation='relu'))
        # cnn_model.add(BatchNormalization())
        cnn_model.add(Conv1D(16, kernel_size=5, activation='relu'))
        cnn_model.add(BatchNormalization())
        cnn_model.add(Conv1D(8, kernel_size=5, activation='relu'))
        cnn_model.add(BatchNormalization())
        cnn_model.add(Conv1D(4, kernel_size=5, activation='relu'))
        cnn_model.add(BatchNormalization())
        cnn_model.add(Conv1D(2, kernel_size=4, activation='relu'))
        cnn_model.add(BatchNormalization())
        cnn_model.add(Conv1D(1, kernel_size=3, activation='relu'))
        cnn_model.add(BatchNormalization())
        # cnn_model.add(Conv1DTranspose(1, kernel_size=(3, 3), activation='relu'))
        # cnn_model.add(BatchNormalization())
        cnn_model.add(Reshape((32,32)))
        cnn_model.summary()

        # cnn_input = Input(shape=uav_data[0].shape)
        # cnn_output = TimeDistributed(cnn_model)(cnn_input)

        # self.model = Model(inputs=cnn_input, outputs=cnn_output)

        # self.y_train = y_train
        # self.x_train = x_train
        # self.y_test = y_test
        # self.x_test = x_test


    def configure(self):
        self.model.compile(
            optimizer='adadelta',
            loss='mean_squared_error',
            metrics=[metrics.mae]
        )

    def train(self):
        self.configure()
        y_train = self.y_train
        x_train = self.x_train
        y_test = self.y_test
        x_test = self.x_test

        callbacks = []
        callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join("checkpoints","cnn","uav-{epoch:02d}-{val_mean_absolute_error:.2f}.hdf5"),
                monitor='val_mean_absolute_error',
                mode='min',
                save_best_only=True,
                save_weights_only=True,
                verbose=True
            )
        )
        
        self.model.fit(x_train, y_train,
                    epochs=self.epics,
                    batch_size=32,
                    shuffle=True,
                    validation_data=(x_test, y_test),
                    callbacks=callbacks)


    def lstmPredict(self, ckpt):
        y_test = self.y_test
        x_test = self.x_test
        self.model.load_weights('checkpoints/cnn/{0}.hdf5'.format(ckpt))
        self.configure()
        prediction = self.model.predict(x_test)
        
        prediction = np.round(np.clip(prediction, 0, 1))

        np.save('data/cnn/prediction.npy', prediction)
        np.save('data/cnn/y_test.npy', y_test)


CSM = Cnn_Model("data/cnnTrainingSets.npy", "data/groundTruths_diff.npy", 3)
CSM.train()
# CSM.meanDensityMap('uav-49-0.82')

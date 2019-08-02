import os
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, LSTM, Conv3DTranspose
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

class Cnn_Lstm_Model:
    def __init__(self, trs=None, grt=None, epics=1):
        self.model = None
        self.epics = epics

        uav_data = np.load(trs)
        print('uav_data: ', uav_data.shape) # (10000, 30, 32, 32, 4)

        uav_label = np.load(grt)
        print('uav_label: ', uav_label.shape) # (10000, 30, 32, 32)

        data_size = int(len(uav_data) * 0.85)

        # (x_train, y_train) = uav_data[:850], uav_label[:850]
        # (x_test, y_test) = uav_data[850:], uav_label[850:]
        (x_train, y_train) = uav_data[:data_size], uav_label[:data_size]
        (x_test, y_test) = uav_data[data_size:], uav_label[data_size:]


        cnn_model = Sequential()
        cnn_model.add(Conv2D(8, kernel_size=(2, 2),
                        activation='relu',
                        input_shape=(32, 32, 4)))
        cnn_model.add(Conv2D(16, kernel_size=(2, 2), activation='relu'))
        cnn_model.add(MaxPooling2D(pool_size=(2,2)))
        cnn_model.add(Conv2D(32, kernel_size=(2, 2), activation='relu'))
        cnn_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        cnn_model.add(MaxPooling2D(pool_size=(2,2)))
        cnn_model.add(Flatten())
        cnn_model.summary()

        # (30*1024) = 2^15, 16384 = 2^14, 4096 = 2^12, 2014 = 2^10 
        lstm_model = Sequential()
        lstm_model.add(LSTM(2048, input_shape=(30, 2304), dropout=0.0, return_sequences=True))
        lstm_model.add(TimeDistributed(Dense(1024)))
        lstm_model.add(TimeDistributed(Reshape((32, 32))))
        lstm_model.summary()

        # upsample_model = Sequential()
        # upsample_model.add(Reshape((16, 8, 8, 1), input_shape=(1, 1024)))
        # upsample_model.add(Conv3DTranspose(2, kernel_size=(4, 3, 3), activation='relu'))
        # upsample_model.add(BatchNormalization())
        # upsample_model.add(Conv3DTranspose(4, kernel_size=(5, 3, 3), activation='relu'))
        # upsample_model.add(BatchNormalization())
        # upsample_model.add(Conv3DTranspose(2, kernel_size=(4, 3, 3), activation='relu'))
        # upsample_model.add(BatchNormalization())
        # upsample_model.add(Conv3DTranspose(1, kernel_size=(5, 3, 3), activation='relu'))
        # upsample_model.add(BatchNormalization())
        # upsample_model.add(Reshape((30, 16, 16)))
        # upsample_model.summary()

        # cnn_input = (?, 30, 32, 32, 4)
        cnn_input = Input(shape=uav_data[0].shape)
        print('input shape: ',cnn_input.shape) # (?, 30, 16, 16, 4)
        lstm_input = TimeDistributed(cnn_model)(cnn_input)
        lstm_output = lstm_model(lstm_input)

        self.model = Model(inputs=cnn_input, outputs=lstm_output)
        self.y_train = y_train
        self.x_train = x_train
        self.y_test = y_test
        self.x_test = x_test

    def configure(self):
        def weighted_binary_crossentropy(weights):
            def w_binary_crossentropy(y_true, y_pred):
                return tf.keras.backend.mean(tf.nn.weighted_cross_entropy_with_logits(
                    y_true,
                    y_pred,
                    weights,
                    name=None
                ), axis=-1)
            return w_binary_crossentropy

        def recall(y_true, y_pred):
            y_true = math_ops.cast(y_true, 'float32')
            y_pred = math_ops.cast(y_pred, 'float32')
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall
        
        def mean_squared_logarithmic_error(y_true, y_pred):
            first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
            second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
            return K.mean(K.square(first_log - second_log), axis=-1)
        
        def sparse_categorical_crossentropy(y_true, y_pred):
            y_true = tf.cast(y_true, 'float32')
            y_pred = tf.cast(y_pred, 'float32')
            return losses.sparse_categorical_crossentropy(y_true, y_pred)

        def kullback_leibler_divergence(y_true, y_pred):
            y_true = tf.cast(y_true, 'float32')
            y_pred = tf.cast(y_pred, 'float32')
            return losses.kullback_leibler_divergence(y_true, y_pred)

        self.model.compile(
            optimizer='adadelta',
            loss=weighted_binary_crossentropy(8.7),
            metrics=[recall]
            # loss='mean_squared_error',
            # metrics=[metrics.mae]
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
                filepath=os.path.join("checkpoints","uav-{epoch:02d}-{val_recall:.2f}.hdf5"),
                monitor='val_recall',
                # filepath=os.path.join("checkpoints","uav-{epoch:02d}-{val_mean_absolute_error:.2f}.hdf5"),
                # monitor='val_mean_absolute_error',
                mode='max',
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
    

    def predict(self, ckpt, index, num):
        y_test = self.y_test
        x_test = self.x_test
        self.model.load_weights('checkpoints/{0}.hdf5'.format(ckpt))
        self.configure()
        prediction = self.model.predict(x_test)

        p = np.round(prediction)

        l = len(y_test[index])
        if l <= num or num < 0:
            num = l

        import cv2
        for i in range(num):
            cv2.imwrite('img/y{0}.png'.format(i), y_test[index][i] * 255)
            cv2.imwrite('img/p{0}.png'.format(i), p[index][i] * 255)
    

    def meanDensityMap(self, ckpt):
        y_test = self.y_test
        x_test = self.x_test
        self.model.load_weights('checkpoints/{0}.hdf5'.format(ckpt))
        self.configure()
        prediction = self.model.predict(x_test)
        
        # y_test/ prediction : (1500, 30, 16, 16)
        prediction = np.round(np.clip(prediction, 0, 1))

        # p = np.sum(prediction, axis=1)
        # p = p / prediction.shape[1]
        # y = np.sum(y_test, axis=1)
        # y = y / y_test.shape[1]

        # for i in range(len(p)):
        #     p[i] = (p[i] - np.min(p[i])) / (np.max(p[i]) - np.min(p[i]))

        # for i in range(len(y)):
        #     y[i] = (y[i] - np.min(y[i])) / (np.max(y[i]) - np.min(y[i]))
        
        # np.save('data/prediction.npy', p)
        # np.save('data/y_test.npy', y)
        np.save('data/prediction.npy', prediction)
        np.save('data/y_test.npy', y_test)


    def generateCNNdata(self, ckpt):
        uav_data = np.load("data/trainingSets_diff.npy")
        print('uav_data: ', uav_data.shape) # (10000, 30, 32, 32, 4)

        self.model.load_weights('checkpoints/{0}.hdf5'.format(ckpt))
        self.configure()
        prediction = self.model.predict(uav_data)
        cnnTrainingSets = np.round(np.clip(prediction, 0, 1))

        np.save('data/cnnTrainingSets.npy', cnnTrainingSets)



CSM = Cnn_Lstm_Model("data/trainingSets_diff.npy", "data/groundTruths_diff.npy", 3)
CSM.train()
# CSM.predict('uav-02-1.00', 0, -1)
# CSM.meanDensityMap('uav-49-0.82')



import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, LSTM,  Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import Flatten, Activation, Reshape
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, TimeDistributed
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras import losses, metrics
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class SumLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(SumLayer, self).__init__()
        self.num_outputs = num_outputs
    
    def call(self, input):
        return K.sum(input, axis=1)

class Lstm_Cnn_Model:
    def __init__(self, trs=None, grt=None, epics=1, weight=1):
        self.model = None
        self.epics = epics
        self.weight = weight

    def loadData(self, trs=None, grt=None):
        uav_data = None
        uav_label = None

        print('\n----data shape----')
        uav_data = np.load(trs)
        uav_label = np.load(grt)
        print('uav_data: ', uav_data.shape) # (10000, 30, 32, 32, 4)
        print('uav_label: ', uav_label.shape) # (10000, 30, 32, 32)

        data_size = int(len(uav_data) * 0.8)
        (x_train, y_train) = uav_data[:data_size], uav_label[:data_size]
        (x_test, y_test) = uav_data[data_size:], uav_label[data_size:]
        
        self.y_train = y_train
        self.x_train = x_train
        self.y_test = y_test
        self.x_test = x_test


    def lstmLayers(self):
        cnn_model = Sequential()
        # cnn_model.add(Conv2D(8, kernel_size=(5, 5),
        #                 activation='relu',
        #                 input_shape=(100, 100, 2)))
        # cnn_model.add(Conv2D(8, kernel_size=(5, 5), activation='relu'))
        # cnn_model.add(MaxPooling2D(pool_size=(2,2)))
        # cnn_model.add(BatchNormalization())
        # cnn_model.add(Conv2D(16, kernel_size=(4, 4), activation='relu'))
        # cnn_model.add(Conv2D(16, kernel_size=(4, 4), activation='relu'))
        # cnn_model.add(MaxPooling2D(pool_size=(2,2)))
        # cnn_model.add(BatchNormalization())
        # cnn_model.add(Conv2D(32, kernel_size=(4, 4), activation='relu'))
        # cnn_model.add(BatchNormalization())
        # cnn_model.add(MaxPooling2D(pool_size=(2,2)))
        # cnn_model.add(BatchNormalization())
        # cnn_model.add(Conv2D(64, kernel_size=(4, 4), activation='relu'))
        # cnn_model.add(BatchNormalization())
        # cnn_model.add(Flatten())
        # cnn_model.summary()
        
        
        cnn_model.add(Flatten(input_shape=(100, 100, 2)))
        cnn_model.add(Dense(16384))
        cnn_model.add(Dense(8192))
        cnn_model.add(Dense(4096))
        cnn_model.add(Dense(2048))
        cnn_model.add(Dense(1024))
        # cnn_model.summary()
        

        # (30*1024) = 2^15, 16384 = 2^14, 4096 = 2^12, 2014 = 2^10 
        lstm_model = Sequential()
        lstm_model.add(LSTM(2048, input_shape=(24, 1024), dropout=0.0, return_sequences=True))
        lstm_model.add(TimeDistributed(Dense(4096)))
        lstm_model.add(TimeDistributed(Dense(8192)))
        cnn_model.add(BatchNormalization())
        lstm_model.add(TimeDistributed(Dense(10000)))
        cnn_model.add(BatchNormalization())
        lstm_model.add(TimeDistributed(Reshape((100, 100))))
        # lstm_model.summary()

        cnn_input = Input(shape=(24, 100, 100, 2))
        lstm_input = TimeDistributed(cnn_model)(cnn_input)
        lstm_output = lstm_model(lstm_input)

        self.model = Model(inputs=cnn_input, outputs=lstm_output)
        

    def cnnLayer(self):
        cnn_model = Sequential()
        cnn_model.add(Reshape((32,32,1), input_shape=(32, 32)))
        cnn_model.add(Conv2D(8, kernel_size=(3,3), activation='relu',))
        cnn_model.add(Conv2D(16, kernel_size=(3,3), activation='relu'))
        cnn_model.add(MaxPooling2D(pool_size=(2,2)))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(3136))
        cnn_model.add(Reshape((14, 14, 16)))
        cnn_model.add(UpSampling2D(size=(2,2)))
        cnn_model.add(Conv2DTranspose(8, kernel_size=(3, 3), activation='relu'))
        cnn_model.add(BatchNormalization())
        cnn_model.add(Conv2DTranspose(1, kernel_size=(3, 3), activation='relu'))
        cnn_model.add(BatchNormalization())
        cnn_model.add(Reshape((32,32)))
        cnn_model.summary()

        cnn_input = Input(shape=(32,32))
        print('input shape: ',cnn_input.shape) # (?, 30, 16, 16, 4)
        cnn_output = cnn_model(cnn_input)

        self.model = Model(inputs=cnn_input, outputs=cnn_output)


    def configure(self, mode='mse'):
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
            y_true = tf.cast(y_true, 'float32')
            y_pred = tf.cast(y_pred, 'float32')
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        if mode=='mse':
            self.model.compile(
                optimizer='adadelta',
                loss='mean_squared_error',
                metrics=[metrics.mae]
            )
        elif mode=='recall':
            self.model.compile(
                optimizer='adadelta',
                loss=weighted_binary_crossentropy(self.weight),
                metrics=[tf.keras.metrics.Recall()]
            )


    def train(self, mode='mse'):
        self.configure(mode)
        y_train = self.y_train
        x_train = self.x_train
        y_test = self.y_test
        x_test = self.x_test
        mc = None

        if mode=='mse':
            mc = ModelCheckpoint(
                filepath=os.path.join("checkpoints","mse-{epoch:02d}-{val_mean_absolute_error:.2f}.hdf5"),
                monitor='val_mean_absolute_error',
                mode='min',
                save_best_only=True,
                save_weights_only=True,
                verbose=True
            )
        else:
            mc = ModelCheckpoint(
                filepath=os.path.join("checkpoints","uav-{epoch:02d}-{val_recall:.2f}.hdf5"),
                monitor='val_recall',
                mode='max',
                save_best_only=True,
                save_weights_only=True,
                verbose=True
            )

        callbacks = []
        callbacks.append(mc)
        
        self.model.fit(x_train, y_train,
                    epochs=self.epics,
                    batch_size=16,
                    shuffle=True,
                    validation_data=(x_test, y_test),
                    callbacks=callbacks)


    def imageData(self, ckpt, x, y, mode='mse', isRound=False, save=False):
        if '.npy' in x:
            x = np.load(x)
        if '.npy' in y:
            y = np.load(y)
        self.model.load_weights('{0}.hdf5'.format(ckpt))
        self.configure(mode)
        result = self.model.evaluate(x, y, batch_size=1)
        print('evaluate result: ', result[1])
        prediction = self.model.predict(x)
        
        if isRound:
            prediction = np.round(np.clip(prediction, 0, 1))
        
        if save:
            np.save('data/prediction.npy', prediction)
            np.save('data/y_test.npy', y)
        
        return prediction, y

    def generateCNN(self):
        x = np.load("data/evaluate_trainingSets.npy")
        self.model.load_weights('checkpoints/uav-01-0.92.hdf5')
        self.configure('recall')
        prediction = self.model.predict(x)
        prediction = np.round(np.clip(prediction, 0, 1))
        prediction = np.sum(prediction, axis=1)
        np.save('data/evaluate_lstm.npy', prediction)

    def test(self):
        x = np.load('data/evaluate_lstm.npy')
        y = np.load('data/evaluate_groundTruths.npy')
        print(x.shape)
        print(y.shape)
        self.model.load_weights('checkpoints/cnn-07-0.03.hdf5')
        self.configure('mse')
        result = self.model.evaluate(x, y, batch_size=1)
        print('result: ', result)
        prediction = self.model.predict(x)
        # prediction = np.round(np.clip(prediction, 0, 1))
        np.save('data/evaluate_cnn.npy', prediction)
        np.save('data/y_test.npy', y)
    


if __name__ == "__main__":
    CSM = Lstm_Cnn_Model(
        epics=10,
        weight=6.44
    )
    CSM.loadData(
        '../../../data/zzhao/uav_regression/cnn/training_data_trajectory.npy',
        '../../../data/zzhao/uav_regression/cnn/training_label_density.npy',
    )
    # CSM.layers()
    CSM.lstmLayers()
    # CSM.cnnLayer()
    CSM.train(mode='mse')
    # CSM.generateCNN()
    # CSM.test()
    # CSM.imageData(
    #     path='../../wbai03/test_postprocess',
    #     ckpt='uav-01-0.91'
    # )
    # CSM.generateCNN()

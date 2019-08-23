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
        self.trainingSets = "data/trainingSets_diff.npy"
        self.grourndTruth = "data/groundTruths_diff.npy"

    def loadData(self, trs=None, grt=None):
        uav_data = None
        uav_label = None

        if trs == None:
            uav_data = np.load(self.trainingSets)
        else:
            uav_data = np.load(trs)
        print('uav_data: ', uav_data.shape) # (10000, 30, 32, 32, 4)

        if grt == None:
            uav_label = np.load(self.grourndTruth)
        else:
            uav_label = np.load(grt)
        print('uav_label: ', uav_label.shape) # (10000, 30, 32, 32)

        data_size = int(len(uav_data) * 0.85)
        (x_train, y_train) = uav_data[:data_size], uav_label[:data_size]
        (x_test, y_test) = uav_data[data_size:], uav_label[data_size:]

        self.y_train = y_train
        self.x_train = x_train
        self.y_test = y_test
        self.x_test = x_test

    def layers(self):

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
        lstm_model.add(LSTM(2048, input_shape=(100, 2304), dropout=0.0, return_sequences=True))
        lstm_model.add(TimeDistributed(Dense(1024)))
        lstm_model.add(TimeDistributed(Reshape((32, 32))))
        lstm_model.add(SumLayer((32, 32)))
        lstm_model.add(BatchNormalization())
        lstm_model.summary()
        
        lstm_model.add(Reshape((32,32,1)))
        lstm_model.add(Conv2D(8, kernel_size=(3,3), activation='relu',))
        lstm_model.add(Conv2D(16, kernel_size=(3,3), activation='relu'))
        lstm_model.add(MaxPooling2D(pool_size=(2,2)))
        lstm_model.add(Flatten())
        lstm_model.add(Dense(3136))
        lstm_model.add(Reshape((14, 14, 16)))
        lstm_model.add(UpSampling2D(size=(2,2)))
        lstm_model.add(Conv2DTranspose(8, kernel_size=(3, 3), activation='relu'))
        lstm_model.add(BatchNormalization())
        lstm_model.add(Conv2DTranspose(1, kernel_size=(3, 3), activation='relu'))
        lstm_model.add(BatchNormalization())
        lstm_model.add(Reshape((32,32)))
        lstm_model.summary()

        cnn_input = Input(shape=(100,32,32,4))
        print('input shape: ',cnn_input.shape) # (?, 30, 16, 16, 4)
        lstm_input = TimeDistributed(cnn_model)(cnn_input)
        lstm_output = lstm_model(lstm_input)

        self.model = Model(inputs=cnn_input, outputs=lstm_output)


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
            y_true = tf.cast(y_true, 'float32')
            y_pred = tf.cast(y_pred, 'float32')
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        self.model.compile(
            optimizer='adadelta',
            # loss=weighted_binary_crossentropy(self.weight),
            # metrics=[recall]
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
                # filepath=os.path.join("checkpoints","uav-{epoch:02d}-{val_recall:.2f}.hdf5"),
                # monitor='val_recall',
                # mode='max',
                filepath=os.path.join("checkpoints","uav-{epoch:02d}-{val_mean_absolute_error:.2f}.hdf5"),
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


    def imageData(self, ckpt, path=None):
        if path == None:
            self.model.load_weights('checkpoints/{0}.hdf5'.format(ckpt))
        else:
            self.model.load_weights('{0}/{1}.hdf5'.format(path, ckpt))
        self.configure()
        prediction = self.model.predict(self.x_test)
        
        np.save('data/prediction.npy', prediction)
        np.save('data/y_test.npy', self.y_test)



CSM = Lstm_Cnn_Model(
    # "data/trainingSets_diff.npy",
    # "data/groundTruths_diff.npy",
    epics=3
)
CSM.loadData(
    "../../wbai03/UAV_POSTPROCESS/data/trainingSets_diff.npy",
    "../../wbai03/UAV_POSTPROCESS/data/groundTruths_diff.npy"
)
CSM.layers()
# CSM.train()
CSM.imageData(
    path='../../wbai03/UAV_POSTPROCESS/checkpoints',
    ckpt='uav-02-0.11'
)
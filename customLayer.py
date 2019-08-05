import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, TimeDistributed


class MyDenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(MyDenseLayer, self).__init__()
    self.num_outputs = num_outputs

  def build(self, input_shape):
    # self.kernel = self.add_variable("kernel",
    #                                 shape=[int(input_shape[-1]),
    #                                        self.num_outputs])

    self.kernel = self.add_variable("kernel",shape=self.num_outputs)

  def call(self, input):
    # return tf.matmul(input, self.kernel)
    return tf.round(tf.clip_by_value(input, 0, 1))

layer = MyDenseLayer((3,2,2))
print(layer(tf.zeros([10, 5])))
print(layer.trainable_variables)



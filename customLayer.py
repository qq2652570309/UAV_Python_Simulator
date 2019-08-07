from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras import backend as K
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np


class MyDenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(MyDenseLayer, self).__init__()
    self.num_outputs = num_outputs
    print('************ num_outputs: ', self.num_outputs)


  def call(self, input):
    return K.sum(input, axis=1)

layer = MyDenseLayer((2, 2))
print(layer(tf.zeros([5, 3, 2, 2])))
print(layer.trainable_variables)
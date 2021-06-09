"""
Filename:           model_inception.py
File Description:   An example of non-sequential deep learning model is Inception v1. A simple DNN model with inception module is created.
                    The model can be retrieved using get_model(). One must specify the input shape and number of output class labels for
                    the model creation. 
Created by:         Abirami Ravi - University of Stuttgart (abirami1429@gmail.com)
References:         Szegedy, Christian, et al. "Going deeper with convolutions."
                    Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate

class Model_Inception:
    def __init__(self):

    # function for creating an inception module
    def __inception_module(self, layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
        # 1x1 conv
        conv1 = Conv2D(f1, (1,1), padding='same', activation='relu', name = 'conv2d_1')(layer_in)

        # 3x3 conv
        conv3 = Conv2D(f2_in, (1,1), padding='same', activation='relu', name = 'conv2d_2')(layer_in)
        conv3 = Conv2D(f2_out, (3,3), padding='same', activation='relu', name = 'conv2d_3')(conv3)

        # 5x5 conv
        conv5 = Conv2D(f3_in, (1,1), padding='same', activation='relu', name='conv2d_4')(layer_in)
        conv5 = Conv2D(f3_out, (5,5), padding='same', activation='relu', name='conv2d_5')(conv5)

        # 3x3 max pooling
        pool = MaxPooling2D((3,3), strides=(1,1), padding='same', name='pool_1')(layer_in)
        pool = Conv2D(f4_out, (1,1), padding='same', activation='relu', name='conv2d_6')(pool)

        # concatenate filters, assumes filters/channels last
        layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1, name='concatenate')
        return layer_out

    def get_model(self, input_shape, no_classes):
        # define model input
        visible = Input(shape=input_shape, name='input_3')

        # add inception block 1
        layer = self.__inception_module(visible, 64, 96, 128, 16, 32, 32)

        #flatten layer
        flatten = tf.keras.layers.Flatten(name='flatten_1')(layer)

        dense_2 = tf.keras.layers.Dense(no_classes, name='dense_1')(flatten)
        output = tf.keras.layers.Softmax(name='softmax_1')(dense_2)

        # create model
        inception_model = Model(inputs=visible, outputs=output)
        inception_model.summary()

        return inception_model
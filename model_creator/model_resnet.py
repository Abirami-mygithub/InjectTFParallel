"""
Filename:           model_resnet.py
File Description:   An example of non-sequential deep learning model is ResNet. ResNet50 pretrained model from keras is used here.
                    The model can be retrieved using get_model(). One must specify the input shape and number of output class labels for
                    the model creation. 
Created by:         Abirami Ravi - University of Stuttgart (abirami1429@gmail.com)
References:         He, Kaiming, et al. "Deep residual learning for image recognition."
                    Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
"""

import tensorflow as tf
from tf.keras.applications import ResNet50
from tf.keras.layers import Flatten
from tf.keras.layers import Dense
from tf.keras import Model

def get_model(shape, no_classes):
    
    resnet = ResNet50(weights='imagenet', include_top=False, input_shape=shape)
    x = Flatten(input_shape=resnet.output.shape)(resnet.output)
    x = Dense(1024, activation='sigmoid')(x)
    predictions = Dense(no_classes, activation='softmax', name='pred')(x)
    resnet_model = Model(inputs=[resnet.input], outputs=[predictions])
    resnet_model.summary()

    return resnet_model

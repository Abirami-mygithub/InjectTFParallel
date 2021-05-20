import sys

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate

# Add trained_models to python path.
root_path = "/home/abirami_ravi/Custom_Layer_Fault_Injection_Software/"

if root_path not in sys.path:
    sys.path.append(root_path)

from datasets.mnist_dataset import MNIST_Dataset

class Inception_Model:
    def __init__(self):
        self.model = self.get_model()

    # function for creating a projected inception module
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

    def get_model(self):
        # define model input
        visible = Input(shape=(28, 28, 1), name='input_3')
        # add inception block 1
        layer = self.__inception_module(visible, 64, 96, 128, 16, 32, 32)
        #flatten layer
        flatten = tf.keras.layers.Flatten(name='flatten_1')(layer)
        dense_2 = tf.keras.layers.Dense(10, name='dense_1')(flatten)
        output = tf.keras.layers.Softmax(name='softmax_1')(dense_2)
        # create model
        inception_model = Model(inputs=visible, outputs=output)
        inception_model.summary()

        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        inception_model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

        return inception_model


    def train_model(self):
        ds = MNIST_Dataset()

        #Fetch the training dataset and train the model
        train_ds = ds.get_training_ds( )
        self.model.fit(train_ds, epochs=4)

        #Fetch the test dataset and evaluate the trained model
        x_test, y_test = ds.get_test_ds()
        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=2)

    def save_weights(self):
        save_path = "/home/abirami_ravi/Custom_Layer_Fault_Injection_Software/saved_models/mnist_model.h5"
        self.model.save(save_path)

        print(f"Saved model: {save_path}.")

    def get_weights_dic(self):
        #weights = self.model.get_weights()
        layers = self.model.layers

        #create a dictionary of layer names and their corresponding weights
        dict_layer_info = {}
        for layer in layers:
            print("layer", layer.name)
            dict_layer_info[layer.name] = layer.get_weights()
        return dict_layer_info

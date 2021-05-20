import tensorflow as tf
from tensorflow.keras import utils

class MNIST_Dataset:
    def __init__(self):
        Dataset = tf.data.Dataset
        mnist = tf.keras.datasets.mnist

        nb_classes = 10

        # Prepare data set
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0

        self.y_train = utils.to_categorical(self.y_train, nb_classes)
        self.y_test = utils.to_categorical(self.y_test, nb_classes)

        # Add a channels dimension
        self.x_train = self.x_train[..., tf.newaxis].astype("float32")
        self.x_test = self.x_test[..., tf.newaxis].astype("float32")

        self.train_ds = Dataset.from_tensor_slices((self.x_train, self.y_train)).shuffle(10000).batch(32)
        self.test_ds = Dataset.from_tensor_slices((self.x_test, self.y_test)).batch(32)

        print(self.x_train.shape)

    def get_training_ds(self):
        return self.train_ds

    def get_test_ds(self):
        return self.x_test, self.y_test




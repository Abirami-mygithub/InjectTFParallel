"""
Filename:           fault_injection_random_bit.py
File Description:   A custom layer called Fault_Injector_Random_Bit is created. This custom layer takes input from the previous layer that it is connected to and
                    injects fault into random bits. The percentage of fault injection is controlled by a parameter called as probability.
Created by:         Abirami Ravi - University of Stuttgart (abirami1429@gmail.com)
References:         https://keras.io/examples/keras_recipes/antirectifier/
                    https://www.tensorflow.org/tutorials/customization/custom_layers
"""


import tensorflow as tf
import numpy as np

class Fault_Injector_Random_Bit(tf.keras.layers.Layer):
    def __init__(self, probability):
        super(Fault_Injector_Random_Bit, self).__init__()
        self.probability = probability
        self.scale = tf.Variable(100000000000000000.)

    def call(self, inputs):

        #copy the inputs to this layer and also its shape
        temp = inputs
        input_shape = tf.shape(inputs, out_type=tf.int64)

        #Usually the layer input datatype will be float32 or float64. These float values are scaled before converting to integers.
        temp = tf.reshape(temp, [-1]) * self.scale


        #Store the shape of the reshaped and scaled.
        shape_input = tf.shape(temp, out_type=tf.int64)

        #Conversion to integer
        temp_int = tf.cast(temp, dtype=tf.int64)

        #Probability tensor is created with the probability values filled
        probability_tensor = tf.fill(dims= shape_input, value=self.probability)

        #Random probability values are generated
        random_tensor = tf.random.uniform(shape = shape_input, minval=0, maxval=1.0, dtype=inputs.dtype)

        #Resultant tensor is created with values only in places where probability value is greater than the random value
        resultant_tensor = tf.math.greater(probability_tensor, random_tensor, name=None)
        resultant_tensor = tf.cast(resultant_tensor,dtype=tf.int64)

        random_bit = np.random.randint(0, 64)
        random_bit_tensor = tf.cast(tf.fill(dims=shape_input, value=random_bit), dtype=tf.int64)

        left_shift_result = tf.bitwise.left_shift(resultant_tensor, random_bit_tensor)

        output = tf.bitwise.bitwise_xor(temp_int, left_shift_result)

        float_output = tf.cast(output, dtype=inputs.dtype)
        #print("float_output", float_output)
        output = float_output /self.scale
        output = tf.reshape(output, shape=input_shape)
        return output

    def get_config(self):
        return {"random_bit_tensor": random_bit_tensor, "probability": self.probability, "name": "fault_injector_random_bit"}
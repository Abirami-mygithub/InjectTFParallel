"""
Filename:           fault_injection_specific_bit.py 
File Description:   A custom layer called Fault_Injector_Specific_Bit is created. This custom layer takes input from the previous layer that it is connected to and 
                    injects fault into specific bit numbers. The bit number is mentioned as a parameter while creating custom layer.
                    Additionally, the percentage of fault injection is controlled by a parameter called as probability. 
Created by:         Abirami Ravi - University of Stuttgart (abirami1429@gmail.com)
"""

import tensorflow as tf

class Fault_Injector_Specific_Bit(tf.keras.layers.Layer):
    def __init__(self, probability, bit_number):
        super(Fault_Injector_Specific_Bit, self).__init__()
        self.probability = probability
        self.scale = tf.Variable(100000000000000000.)
        self.bit_number = bit_number

    def call(self, inputs):

      if (inputs.dtype == tf.float32):
            input_datatype = tf.float32
        else:
            input_datatype = tf.float64

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
        random_tensor = tf.random.uniform(shape = shape_input, minval=0, maxval=1.0, dtype=input_datatype)

        #Resultant tensor is created with values only in places where probability value is greater than the random value
        resultant_tensor = tf.math.greater(probability_tensor, random_tensor, name=None)
        resultant_tensor = tf.cast(resultant_tensor,dtype=tf.int64)

        random_bit_tensor = tf.cast(tf.fill(dims=shape_input, value=self.bit_number), dtype=tf.int64)

        left_shift_result = tf.bitwise.left_shift(resultant_tensor, random_bit_tensor)

        output = tf.bitwise.bitwise_xor(temp_int, left_shift_result)

        float_output = tf.cast(output, dtype=input_datatype)
        #print("float_output", float_output)
        output = float_output /self.scale
        output = tf.reshape(output, shape=input_shape)
        return output

    def get_config(self):
        return {"specific_bit_tensor": random_bit_tensor, "probability": self.probability, "name": "fault_injector_specific_bit"}
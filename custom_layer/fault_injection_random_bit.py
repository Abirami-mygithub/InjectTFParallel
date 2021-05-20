import tensorflow as tf
import numpy as np

class Fault_Injector_Random_Bit(tf.keras.layers.Layer):
    def __init__(self, probability):
        super(Fault_Injector_Random_Bit, self).__init__()
        self.probability = probability


    def call(self, inputs):
        print("input tensor:", inputs)

        #To flip random bit, formula is: num xor (1 << random_bit)
        #convert the received input to int32 or int64 to make bit modifications
        if inputs.dtype == tf.float32:
            converted_inputs = tf.cast(inputs, dtype=tf.int32)

            shape_input = tf.shape(inputs, out_type=tf.int32)

            #resultant_tensor = self.__generate_tensor_probability(inputs, data_type =tf.int32)
            #create a tensor of probability values
            probability_tensor = tf.fill(dims= shape_input, value=self.probability)

            #create a random probability tensor
            random_tensor = tf.random.uniform(shape = shape_input, minval=0, maxval=1, dtype=tf.float32)

            #compare probability tensor with the random probability tensor
            resultant_tensor = tf.math.greater(probability_tensor, random_tensor, name=None)
            resultant_tensor = tf.cast(resultant_tensor,dtype=tf.int32)

            #generate random
            random_bit = np.random.randint(0, 32)

        elif inputs.dtype == tf.float64:
            converted_inputs = tf.cast(inputs, dtype=tf.int64)

            #create a tensor of probability values
            probability_tensor = tf.fill(dims= shape_input, value=self.probability)

            #create a random probability tensor
            random_tensor = tf.random.uniform(shape = shape_input, minval=0, maxval=1, dtype=tf.float64)

            #compare probability tensor with the random probability tensor
            resultant_tensor = tf.math.greater(probability_tensor, random_tensor, name=None)
            resultant_tensor = tf.cast(resultant_tensor,dtype=tf.int64)

            #generate random
            random_bit = np.random.randint(0, 64)
        else:
            raise NotImplementedError(
                "Bit flip is not supported for dtype: ", element_val.dtype)

        #create a random bit tensor
        random_bit_tensor = tf.fill(dims=shape_input, value=random_bit)


        left_shift_result = tf.bitwise.left_shift(resultant_tensor, random_bit_tensor)

        output = tf.bitwise.bitwise_xor(converted_inputs, left_shift_result)
        output = tf.cast(output, dtype=inputs.dtype)

        return output

    def get_config(self):
        return {"random_bit_tensor": random_bit_tensor, "probability": self.probability, "name": "fault_injector_random_bit"}

    def __generate_tensor_probability(self, inputs, data_type):

        resultant_tensor = tf.zeros_like(inputs)
        shape_input = tf.shape(inputs, out_type=data_type)
        iter_shape = (32, inputs.shape[1], inputs.shape[2], inputs.shape[3])
        indice = []
        for i in range(32):
            element = []
            random_number = np.random.rand()
        if(self.probability > random_number):
            for dim in iter_shape:
                element.append(np.int32(np.random.randint(0, dim)))
            #print(element)
        zero_np = np.zeros(iter_shape)

        #Initialize with zeros and then append values 1 in the indexes chosen
        resultant_tensor = tf.Variable(zero_np, shape=iter_shape, name="a", dtype=data_type)
        for index in indice:
            zero_np[index] = 1
            resultant_tensor.assign_add(zero_np)

        return resultant_tensor

import tensorflow as tf

class Fault_Injector_Specific_Bit(tf.keras.layers.Layer):
    def __init__(self, specific_bit, probability):
        super(Fault_Injector_Specific_Bit, self).__init__()
        self.specific_bit = specific_bit
        self.probability = probability

    def call(self, inputs):
      print("input tensor:", inputs)

      #To flip specific bit formula is: num xor (1 << specifc_bit)
      #convert the received input to int32 or int64 to make bit modifications
      converted_inputs = tf.cast(inputs, dtype=tf.int32)

      shape_input = tf.shape(inputs, out_type=tf.int32)

      #create a tensor of probability values
      probability_tensor = tf.fill(dims= shape_input, value=self.probability)

      #create a random probability tensor
      random_tensor = tf.random.uniform(shape = shape_input, minval=0, maxval=1, dtype=tf.float32)

      #compare probability tensor with the random probability tensor
      resultant_tensor = tf.math.greater(probability_tensor, random_tensor, name=None)
      resultant_tensor = tf.cast(resultant_tensor,dtype=tf.int32)

      #create a specific bit tensor
      specific_bit_tensor = tf.fill(dims=shape_input, value=self.specific_bit)


      left_shift_result = tf.bitwise.left_shift(resultant_tensor, specific_bit_tensor)

      output = tf.bitwise.bitwise_xor(converted_inputs, left_shift_result)
      output = tf.cast(output, dtype=inputs.dtype)

      return output

    def get_config(self):
        return {"specific_bit": self.specific_bit, "probability": self.probability, "name": "fault_injector_specific_bit"}
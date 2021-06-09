"""
Filename:           trainer_inception.py 
File Description:   The Trainer_Inception class fetches the dataset, inception model architecture, compiles and trains the model. 
                    Also the trained model weights are saved in .h5 format. The only accessible function from this class is get_weights_dic() 
                    through which model's weights can be obtained in dictionary format with layers and corresponding weights.
Created by:         Abirami Ravi - University of Stuttgart (abirami1429@gmail.com)
References:         https://keras.io/examples/keras_recipes/antirectifier/
                    https://www.tensorflow.org/tutorials/customization/custom_layers
"""

import sys
import os.path
from tensorflow.keras.losses import CategoricalCrossentropy


# Add trained_models to python path.
root_path = "/home/abirami_ravi/Custom_Layer_Fault_Injection_Software/"

if root_path not in sys.path:
    sys.path.append(root_path)

from datasets.mnist_dataset import MNIST_Dataset
from model_creator.model_inception import Model_Inception

class Trainer_Inception:
    def __init__(self):
    
    def __train_model(self):
        #Fetch dataset
        self.train_dataset = MNIST_Dataset.get_training_ds()

        #Fetch the model
        model = Model_Inception.get_model(input_shape= (28, 28, 1), no_classes= 10)

        self.compiled_model = self.__compile_model(model)

        print("******......Model training begins......******")
        #Train the model
        self.compiled_model.fit(self.train_dataset, epochs=4)

        #Fetch test dataset and evaluate the trained model
        x_test, y_test = MNIST_Dataset.get_test_ds()
        test_loss, test_acc = self.compiled_model.evaluate(x_test, y_test, verbose=2)

        print("loss:", test_loss)
        print("accuracy:", test_acc)
    
        #Kindly modify the path for saving the model
        save_path = "/home/abirami_ravi/Custom_Layer_Fault_Injection_Software/saved_models/mnist_model.h5"
        self.compiled_model.save(save_path)
        print(f"Saved model: {save_path}.")

        return self.compiled_model

    def __compile_model(self, model_to_be_compiled):
        loss_fn = CategoricalCrossentropy(from_logits=True)
        model_to_be_compiled.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
        return model_to_be_compiled
        

    def get_weights_dic(self):
        
        if os.path.isfile(save_path):
            print ("******......Saved model exists......******")
            loaded_model = keras.models.load_model(save_path)
            constructed_model = self.__compile_model(loaded_model)
        else:
            print ("*****......Saved model does not exist*****")
            self.__train_model()
            constructed_model = self.compiled_model
        
        print("******......Fetching weights from trained model......******")
        layers = constructed_model.layers

        #create a dictionary of layer names and their corresponding weights
        dict_layer_info = {}
        for layer in layers:
            print("layer", layer.name)
            dict_layer_info[layer.name] = layer.get_weights()

        return dict_layer_info

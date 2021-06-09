"""
Filename:           fault_injector.py
File Description:   This file is responsible for fetching the model architecture based on user's choice, reading fault specification
                    from configuration file, creating appropriate custom layer based on fault type, modifying the original model
                    with custom layer inserted for fault injection, thereby creating a new fault injected model
                    and evaluating the fault injected model.
Created by:         Abirami Ravi - University of Stuttgart (abirami1429@gmail.com)
References:         https://www.tensorflow.org/api_docs/python/tf/keras/models/model_from_yaml
                    https://www.educative.io/edpresso/keras-load-save-model
"""

import yaml
import sys
import random
import copy
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy


# Add root folder to python path.
root_path = "/home/abirami_ravi/Custom_Layer_Fault_Injection_Software/InjectTFParallel/"

if root_path not in sys.path:
    sys.path.append(root_path)


from model_trainer.trainer_inception import Trainer_Inception
from custom_layer.fault_injection_random_bit import Fault_Injector_Random_Bit
from custom_layer.fault_injection_specific_bit import Fault_Injector_Specific_Bit
import constants as cts
import config_manager
from evaluation.evaluate import Evaluation

class Fault_Injector:

    def __init__(self):
        self.cm = config_manager.ConfigurationManager(path_to_config_file=cts.CONFIGURATION_PATH)

    def inject_fault(self):
        print("Fault injection is available for following models:")
        print("1. User defined model with inception module trained on mnist dataset")
        print("2. ResNet50 trained on GTSRB")
        chosen_option = input("Kindly choose one of the above for fault injection:")
        chosen_option = int(chosen_option)
        if(chosen_option == cts.Available_Models.INCEPTION_MODEL.value):
            self.__fault_injector_selected_model(chosen_option)
        elif(chosen_option == cts.Available_Models.RESNET50_MODEL.value):
            self.__fault_injector_selected_model(chosen_option)
        else:
            print("Selected option does not exist")

    def __fault_injector_selected_model(self, selected_model):
        if(selected_model == cts.Available_Models.INCEPTION_MODEL.value):
            ti_obj = Trainer_Inception()
            model, self.weights = ti_obj.get_model_and_weights()
            self.model_dict = self.__convert_model_to_yaml(model)
            self.__process_config_data()

    def __convert_model_to_yaml(self, model):
        print("******......Model architecture conversion to yaml......******")
        model_in_yaml_format = model.to_yaml()
        model_in_python_format = yaml.load(model_in_yaml_format)

        return model_in_python_format

    def __process_config_data(self):
        config_data = self.cm.get_data()
        print("******......Fetching fault injection configuration data......******")
        for count, elements in enumerate(config_data):
            new_instance = self.__create_new_fault_injector_instance()
            self.__custom_layer_creator(count, elements)

    def __custom_layer_creator(self, index_position, individual_fault_spec):

        print("******......Custom layer creation......******")

        if((individual_fault_spec[cts.FAULT_TYPE_STR] == cts.BIT_FLIP_STR) and
           (individual_fault_spec[cts.BIT_FLIP_TYPE_STR]== cts.SPECIFIC_BIT_STR)):
            custom_layer= {'class_name': cts.SPECIFIC_BIT_FAULT_INJECTOR,
                            'config': {'probability': individual_fault_spec[cts.PROBABILITY_STR],
                                       'bit_number': individual_fault_spec[cts.BIT_NUMBER_STR]},
                            'inbound_nodes': [[[str(individual_fault_spec[cts.LAYER_NAME_STR]), 0, 0, {}]]],
                            'name': 'fault__injector__specific__bit'+ str(index_position+1)}


        elif((individual_fault_spec[cts.FAULT_TYPE_STR] == cts.BIT_FLIP_STR) and
           (individual_fault_spec[cts.BIT_FLIP_TYPE_STR]== cts.RANDOM_BIT_STR)):
            custom_layer = {'class_name': cts.RANDOM_BIT_FAULT_INJECTOR,
                            'config': {'probability': individual_fault_spec[cts.PROBABILITY_STR]},
                            'inbound_nodes': [[[str(individual_fault_spec[cts.LAYER_NAME_STR]), 0, 0, {}]]],
                            'name': 'fault__injector__random__bit'+ str(index_position+1)}

        print(custom_layer)
        fault_type = individual_fault_spec[cts.BIT_FLIP_TYPE_STR]
        self.__custom_layer_insertion(custom_layer, individual_fault_spec[cts.LAYER_NAME_STR], fault_type)

    def __custom_layer_insertion(self, custom_layer, selected_layer, fault_type):

        print("******......New model creation with custom layer......******")

        new_model = copy.deepcopy(self.model_dict)
        layers_info = new_model['config']['layers']
        is_layer_inserted = False

        custom_layer_type = custom_layer['class_name']
        for index, layer in enumerate(layers_info):
            if(layer['name'] == selected_layer):
                if(is_layer_inserted == False):
                    layers_info.insert(1, custom_layer)
                    is_layer_inserted = True
                    print("layer inserted")
            if(layer['class_name']!= 'InputLayer' and ((layer['class_name']!= (cts.SPECIFIC_BIT_FAULT_INJECTOR))or (layer['class_name']!=(cts.RANDOM_BIT_FAULT_INJECTOR)))):
                for nodes in layer['inbound_nodes'][0]:
                    if(nodes[0] == selected_layer):
                        nodes[0] = custom_layer['name']

        self.__construct_model_from_dic(new_model, selected_layer, custom_layer_type)

    def __construct_model_from_dic(self, fault_model, layer, custom_layer_type):
        config_model = yaml.dump(fault_model)

        if(custom_layer_type == cts.SPECIFIC_BIT_FAULT_INJECTOR):
            fault_injected_model = tf.keras.models.model_from_yaml(config_model, custom_objects={custom_layer_type: Fault_Injector_Specific_Bit})
        else:
            fault_injected_model = tf.keras.models.model_from_yaml(config_model, custom_objects={custom_layer_type: Fault_Injector_Random_Bit})

        fault_injected_model.summary()

        self.__assign_weights_to_customized_model(fault_injected_model)

    def __assign_weights_to_customized_model(self, fault_injected_model):

        print("******......Assigning weights to fault injected model......******")
        slices = fault_injected_model.layers
        for item in slices:
            #print(item.name)
            if "fault__injector__" not in item.name:
                item.set_weights(self.weights[item.name])

        self.__compile_fault_model(fault_injected_model)

    def __compile_fault_model(self, fault_injected_model):

        print("******......Compilation begins......******")
        loss_fn = CategoricalCrossentropy(from_logits=True)
        fault_injected_model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
        self.__evaluate_fault_injected_model(fault_injected_model)

    def __evaluate_fault_injected_model(self, fault_injected_model):

        print("******......Evaluation begins......******")
        ev = Evaluation()
        ev.evaluate_model(fault_injected_model)

    def __create_new_fault_injector_instance(self):
        return Fault_Injector
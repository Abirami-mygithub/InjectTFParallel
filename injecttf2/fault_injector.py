
import yaml
import tensorflow as tf
from tensorflow.keras.utils import plot_model


import sys
# Add trained_models to python path.
root_path = "/home/abirami_ravi/Custom_Layer_Fault_Injection_Software/"

if root_path not in sys.path:
    sys.path.append(root_path)
from models.inception_model import Inception_Model
from custom_layer.fault_injection_random_bit import Fault_Injector_Random_Bit
from custom_layer.fault_injection_specific_bit import Fault_Injector_Specific_Bit
from evaluation.evaluate import Evaluation

class Fault_Injector:
    def __init__(self):
        self.im = Inception_Model()

    def inject_fault(self, model, config_data, weights):
        #convert the model into yaml format
        for data in config_data:
            temp_model = model
            model_config = self.__get_model_in_yaml(temp_model)

            #create custom layer dictionary to insert in model yaml file
            custom_layer, selected_layer = self.__custom_layer_dictionary(data)

            custom_layer_type = custom_layer['class_name']
            #flag to check if the layer is already inserted
            is_layer_inserted = False

            layers_info = model_config['config']['layers']
            for index, layer in enumerate(layers_info):
                if(layer['name'] == selected_layer):
                    if(is_layer_inserted == False):
                        layers_info.insert(1, custom_layer)
                        is_layer_inserted = True
                if(custom_layer_type == 'Fault_Injector_Specific_Bit'):
                    if(layer['class_name']!= 'InputLayer' and layer['class_name']!= 'Fault_Injector_Specific_Bit' ):
                        for nodes in layer['inbound_nodes'][0]:
                            if(nodes[0] == selected_layer):
                                nodes[0] = 'fault__injector__specific__bit'
                elif(custom_layer_type == 'Fault_Injector_Random_Bit'):
                    if(layer['class_name']!= 'InputLayer' and layer['class_name']!= 'Fault_Injector_Random_Bit' ):
                        for nodes in layer['inbound_nodes'][0]:
                            if(nodes[0] == selected_layer):
                                nodes[0] = 'fault__injector__random__bit'
                else:
                    print("Fault_type is invalid")
            mod = self.__construct_model_from_yaml(model_config, custom_layer_type)
            model_with_weights = self.__set_weights(mod, weights)

            loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            model_with_weights.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

            self.__evaluate_fault_injected_model(model_with_weights, custom_layer_type)

    def __evaluate_fault_injected_model(self, model_, fault_type_name):
        ev = Evaluation(model_, fault_type_name)
        ev.plot_confusion_matrix()
        ev.evaluate_model()

    def __set_weights(self, fault_model, weights_from_original_model):
        slices = fault_model.layers
        for item in slices:
            #print(item.name)
            if "fault__injector__" not in item.name:
                item.set_weights(weights_from_original_model[item.name])
        return fault_model


    def __construct_model_from_yaml(self, model, layer_type):
        config = yaml.dump(model)
        #print("modified yaml content:", config)
        print("layer type:", layer_type)
        if(layer_type == 'Fault_Injector_Specific_Bit'):
            mod = tf.keras.models.model_from_yaml(config, custom_objects={layer_type: Fault_Injector_Specific_Bit})
        else:
            mod = tf.keras.models.model_from_yaml(config, custom_objects={layer_type: Fault_Injector_Random_Bit})
        #plot_model(mod, show_shapes=True, to_file= 'Fault_Model.png')
        mod.summary()
        return mod


    def __get_model_in_yaml(self, model):
        yaml_model = model.to_yaml()
        #print("Before:", yaml_model)
        return (yaml.load(yaml_model))

    def __custom_layer_dictionary(self, config_data):
        fault_type = config_data['bit_flip_type']
        layer_name = config_data['layer_name']
        probability = config_data['probability']
        layer_name = config_data['layer_name']
        if(fault_type == "SpecificBit"):
            custom_layer_dict = {'class_name': 'Fault_Injector_Specific_Bit',
                            'config': {'probability': probability, 'specific_bit': config_data['bit_number']},
                            'inbound_nodes': [[[str(layer_name), 0, 0, {}]]],
                            'name': 'fault__injector__specific__bit'}
        elif (fault_type == "RandomBit"):
            custom_layer_dict = {'class_name': 'Fault_Injector_Random_Bit',
                            'config': {'probability': probability},
                            'inbound_nodes': [[[str(layer_name), 0, 0, {}]]],
                            'name': 'fault__injector__random__bit'}
        else:
            print("Fault type is invalid")
        return custom_layer_dict, layer_name

#
# InjectTF2
# A fault injection framework for TensorFlow 2
#
# TU-Dresden, Institute of Automation (IfA)
#

import logging
import yaml

class ConfigurationManager:
    def __init__(self, path_to_config_file="/home/abirami_ravi/Custom_Layer_Fault_Injection_Software/config/config_fault_type_specific_bit.yml"):
        """InjectTF2 configuration class.

        This class handles reading and parsing the yaml configuration file.

        Args:
            path_to_config_file (str): Path to the configuration file. Defaults to
                "./config/InjectTF2_conf.yml".
        """

        self.__config_data = self.read_config(path_to_config_file)

    def read_config(self, file):

        logging.debug("Reading configuration file {0}".format(file))

        file = (
            file if (file.endswith(".yml") or file.endswith(".yaml")) else file + ".yml"
        )

        try:
            with open(file, "rb") as f:
                data = yaml.safe_load(f)
        except IOError as error:
            print("Can not open file: ", file)
            raise

        logging.debug("Done reading config file.\nData is:\n{0}".format(data))

        return data

    def get_data(self):
        """Returns a dictionary containing the complete data of the configuration file."""
        return self.__config_data

    def get_selected_layer(self):
        """Returns the name of the selected layer."""
        return self.__config_data[str_res.layer_name_str]

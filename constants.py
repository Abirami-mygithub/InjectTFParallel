"""
Filename:           constants.py
File Description:   The constants.py consists of path variables, enums which can be modified based on the user.
                    Also contains constants whose value does not change
Created by:         Abirami Ravi - University of Stuttgart (abirami1429@gmail.com)
"""

from enum import Enum

class Available_Models(Enum):
    INCEPTION_MODEL = 1
    RESNET50_MODEL = 2

CONFIGURATION_PATH = "/home/abirami_ravi/Custom_Layer_Fault_Injection_Software/InjectTFParallel/config/Fault_injection_config_file.yml"

SPECIFIC_BIT_FAULT_INJECTOR = 'Fault_Injector_Specific_Bit'
RANDOM_BIT_FAULT_INJECTOR = 'Fault_Injector_Random_Bit'

# key words for config file
LAYER_NAME_STR = "layer_name"
PROBABILITY_STR = "probability"
FAULT_TYPE_STR = "fault_type"
BIT_FLIP_TYPE_STR = "bit_flip_type"
BIT_NUMBER_STR = "bit_number"

# values for `fault_type` key
BIT_FLIP_STR = "BitFlip"

# values for `bit_flip_type` key
RANDOM_BIT_STR = "RandomBit"
SPECIFIC_BIT_STR = "SpecificBit"


GTSRB_LABELS = { 0:'Speed limit (20km/h)',
                1:'Speed limit (30km/h)',
                2:'Speed limit (50km/h)',
                3:'Speed limit (60km/h)',
                4:'Speed limit (70km/h)',
                5:'Speed limit (80km/h)',
                6:'End of speed limit (80km/h)',
                7:'Speed limit (100km/h)',
                8:'Speed limit (120km/h)',
                9:'No passing',
                10:'No passing veh over 3.5 tons',
                11:'Right-of-way at intersection',
                12:'Priority road',
                13:'Yield',
                14:'Stop',
                15:'No vehicles',
                16:'Veh > 3.5 tons prohibited',
                17:'No entry',
                18:'General caution',
                19:'Dangerous curve left',
                20:'Dangerous curve right',
                21:'Double curve',
                22:'Bumpy road',
                23:'Slippery road',
                24:'Road narrows on the right',
                25:'Road work',
                26:'Traffic signals',
                27:'Pedestrians',
                28:'Children crossing',
                29:'Bicycles crossing',
                30:'Beware of ice/snow',
                31:'Wild animals crossing',
                32:'End speed + passing limits',
                33:'Turn right ahead',
                34:'Turn left ahead',
                35:'Ahead only',
                36:'Go straight or right',
                37:'Go straight or left',
                38:'Keep right',
                39:'Keep left',
                40:'Roundabout mandatory',
                41:'End of no passing',
                42:'End no passing veh > 3.5 tons' }

GTSRB_IMG_HEIGHT = 32
GTSRB_IMG_WIDTH = 32
GTSRB_channels = 3


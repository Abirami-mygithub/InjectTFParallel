import sys
# Add trained_models to python path.
root_path = "/home/abirami_ravi/Custom_Layer_Fault_Injection_Software/"

if root_path not in sys.path:
    sys.path.append(root_path)

from models.inception_model import Inception_Model
from injecttf2.config_manager import ConfigurationManager
from injecttf2.fault_injector import Fault_Injector

im = Inception_Model()
model = im.get_model()
weights = im.get_weights_dic() #TBM

model.summary()

cm = ConfigurationManager("/home/abirami_ravi/Custom_Layer_Fault_Injection_Software/config/config_fault_type_random_bit.yml")
config_data = cm.get_data()
print(config_data)

fi = Fault_Injector()
fi.inject_fault(model, config_data, weights)
#function for return fault injection model
#Test and evaluation function
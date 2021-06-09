from config_manager import ConfigurationManager
from fault_injector import Fault_Injector

cm_obj = ConfigurationManager(path_to_config_file="/home/abirami_ravi/Custom_Layer_Fault_Injection_Software/InjectTFParallel/config/config_fault_injection_inception_model.yml")
data = cm_obj.get_data()
print(data)

fi_obj = Fault_Injector()
fi_obj.inject_fault()
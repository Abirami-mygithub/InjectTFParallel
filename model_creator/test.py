from model_inception import Model_Inception
import model_resnet

obj = Model_Inception()

model = obj.get_model(input_shape=(28, 28, 1), no_classes=10)
model.summary()

resnet_mod = model_resnet.get_model(shape=(32,32,3), no_classes=43)
resnet_mod.summary()
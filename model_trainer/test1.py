from trainer_inception import Trainer_Inception

obj = Trainer_Inception()
mod, wts = obj.get_model_and_weights()
mod.summary()
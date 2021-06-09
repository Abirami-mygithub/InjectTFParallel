from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import sys

# Add trained_models to python path.
root_path = "/home/abirami_ravi/Custom_Layer_Fault_Injection_Software/"

if root_path not in sys.path:
    sys.path.append(root_path)

from datasets.mnist_dataset import MNIST_Dataset

class Evaluation:
    def __init__(self):
        ds = MNIST_Dataset()
        self.x_test, self.y_test = ds.get_test_ds()


    #def plot_model_architecture(self):
        #plot_model(self.model, show_shapes=True, to_file=self.name)

    '''def plot_confusion_matrix(self):
        predicted_output = self.model.predict(self.x_test)

        #convert the hot-coded output from the model's prediction to labels
        y_pred = []
        for element in predicted_output:
            y_pred.append(np.argmax(element))

        y_out = []
        for element in self.y_test:
            y_out.append(np.argmax(element))

        label = [0, 1, 2, 3, 4, 5, 6, 7, 8 ,9]

        cm = confusion_matrix(y_out, y_pred, normalize='true', labels=label)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=label)
        fig, ax = plt.subplots(figsize=(10, 10))
        disp.plot(ax=ax)'''

    def evaluate_model(self, model):
        model.evaluate(self.x_test[:100], self.y_test[:100])



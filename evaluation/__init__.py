from tensorflow.keras.utils import plot_model

# plot model architecture
def plot_model_arch(model, name):
    plot_model(model, show_shapes=True, to_file=name)


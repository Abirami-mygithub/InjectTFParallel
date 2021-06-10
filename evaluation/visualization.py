from keract import get_activations, display_activations

def visualize_intermediate_layer_outputs(model):
  keract_inputs = x_test[:1]
  keract_targets = y_test[:1]
  activations = get_activations(model, keract_inputs)
  display_activations(activations, cmap="gray", save=False )

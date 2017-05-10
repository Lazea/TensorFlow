"""Base model class"""

import model_utils

class Model:

    def __init__(self, params_path):
        """Constructor.
        Loads model params"""
        if params_path != None:
            self.params = load_params(params_path)
        else:
            self.params = None

        self.input_size = None
        self.weights = None
        self.biases = None

    def load_params(self, path):
        return model_utils.load_model_params(path)

    def save_params(self, values, path):
        model_utils.save_model_params(values, path)

    def get_weights_and_biases(self, values):
        weights = values[0][0]
        biases = values[0][1]
        return weights, biases

    def create(self):
        pass

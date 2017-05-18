"""Base model class"""

import model_utils

class Model:

    def __init__(self, params_path):
        """Constructor.
        Loads model params"""
        if params_path != None:
            self.params = self.load_params(params_path)
        else:
            self.params = None

        self.input_size = None
        self.x = None
        self.weights = None
        self.biases = None
        self.logits = None
        self.labels = None

    def load_params(self, path):
        return model_utils.load_model_params(path)

    def save_params(self, values, path):
        model_utils.save_model_params(values, path)

    def get_weights_and_biases(self, values):
        """Returns all model weights and biases separately"""
        weights = values[0, :]
        biases = values[1, :]
        return weights, biases

    def create(self, params=None):
        """Base model create function which loads model
        parameters if they exist"""
        # Load model parameters if they exist
        if self.params != None:
            self.weights, self.biases = self.get_weights_and_biases(self.params)

from .Layer import Layer
import numpy as np

class LSTM(Layer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
import sys

import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_weights = None
        self.output_weights = None

    def load_weights(self, weights_file):
        with open(weights_file, 'r') as file:
            lines = file.readlines()

        # Read network structure
        structure = lines[0].split()
        self.input_size = int(structure[0])
        self.hidden_size = int(structure[1])
        self.output_size = int(structure[2])

        # Read hidden weights
        hidden_weights_str = lines[1].strip()[1:-1]
        output_weights_str = lines[2].strip()[1:-1]
        weights_str = hidden_weights_str + ',' + output_weights_str
        weights = [float(w) for w in weights_str.split(',')]

        # Set hidden weights
        hidden_weights_size = self.input_size * self.hidden_size
        self.hidden_weights = np.array(weights[:hidden_weights_size]).reshape(self.input_size, self.hidden_size)

        # Set output weights
        output_weights_size = self.hidden_size * self.output_size
        self.output_weights = np.array(weights[hidden_weights_size:hidden_weights_size+output_weights_size]).reshape(self.hidden_size, self.output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # foward propagation
    def forward(self, inputs):
        hidden_layer = np.dot(inputs, self.hidden_weights)
        hidden_layer_activation = self.sigmoid(hidden_layer)
        output_layer = np.dot(hidden_layer_activation, self.output_weights)
        output = self.sigmoid(output_layer)
        if output > 0.5:
            return 1
        return 0

# write the predicted output to a file
def classify_data(network, data_file, output_file):
    with open(data_file, 'r') as file:
        lines = file.readlines()
    with open(output_file, 'w') as file:
        for line in lines:
            inputs = np.array([int(ch) for ch in line.strip() if ch.isdigit()], dtype=float)
            output = network.forward(inputs)
            file.write(str(output) + '\n')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python runnet0.py <Learning File> <Test File>")
        sys.exit(1)

    network = NeuralNetwork(0, 0, 0)
    network.load_weights(sys.argv[1])
    classify_data(network, sys.argv[2], 'output0.txt')
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
        hidden_weights_str = lines[1].strip()[1:-1]  # Remove brackets and newline
        hidden_weights = [float(w) for w in hidden_weights_str.split(',')]
        self.hidden_weights = np.array(hidden_weights).reshape(self.input_size, self.hidden_size)

        # Read output weights
        output_weights_str = lines[2].strip()[1:-1]  # Remove brackets and newline
        output_weights = [float(w) for w in output_weights_str.split(',')]
        self.output_weights = np.array(output_weights).reshape(self.hidden_size, self.output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        hidden_layer = np.dot(inputs, self.hidden_weights)
        hidden_layer_activation = self.sigmoid(hidden_layer)
        output_layer = np.dot(hidden_layer_activation, self.output_weights)
        output = self.sigmoid(output_layer)
        return output

def classify_data(network, data_file, output_file):
    with open(data_file, 'r') as file:
        lines = file.readlines()

    with open(output_file, 'w') as file:
        for line in lines:
            inputs = np.array([int(ch) for ch in line.strip() if ch.isdigit()], dtype=float)
            output = network.forward(inputs)
            classification = 1 if output > 0.5 else 0
            file.write(str(classification) + '\n')

# Usage example
network = NeuralNetwork(0, 0, 0)
network.load_weights('wnet')
classify_data(network, 'data_file.txt', 'output_file.txt')

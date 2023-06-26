import sys

import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_weights = None
        self.output_weights = None
        # self.hidden_bias = None
        # self.output_bias = None

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
        output_weights_str = lines[2].strip()[1:-1]  # Remove brackets and
        weights_str = hidden_weights_str + ',' + output_weights_str
        weights = [float(w) for w in weights_str.split(',')]
        # self.hidden_bias = weights[-2]
        # self.output_bias = weights[-1]
        # weights = weights[:-2]

        # Set hidden weights
        hidden_weights_size = self.input_size * self.hidden_size
        self.hidden_weights = np.array(weights[:hidden_weights_size]).reshape(self.input_size, self.hidden_size)

        # Set output weights
        output_weights_size = self.hidden_size * self.output_size
        self.output_weights = np.array(weights[hidden_weights_size:hidden_weights_size+output_weights_size]).reshape(self.hidden_size, self.output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
       # print( self.hidden_weights)
        hidden_layer = np.dot(inputs, self.hidden_weights)
        #print(hidden_layer)
        hidden_layer_activation = self.sigmoid(hidden_layer)
        output_layer = np.dot(hidden_layer_activation, self.output_weights)
        output = self.sigmoid(output_layer)
        if output > 0.5:
            return 1
        return 0

def classify_data(network, data_file, output_file):
    with open(data_file, 'r') as file:
        lines = file.readlines()
    with open(output_file, 'w') as file:
        for line in lines:
            inputs = np.array([int(ch) for ch in line.strip() if ch.isdigit()], dtype=float)
            output = network.forward(inputs)
            file.write(str(output) + '\n')

def calculate_accuracy(classification_file, output_file):
    # Read the lines from the classification file
    with open(classification_file, "r") as classification:
        classification_lines = classification.readlines()

    # Read the lines from the output file
    with open(output_file, "r") as output:
        output_lines = output.readlines()

    # Remove any leading or trailing whitespace from the lines
    classification_lines = [line.strip() for line in classification_lines]
    output_lines = [line.strip() for line in output_lines]

    # Calculate the accuracy rate
    total = len(classification_lines)
    correct = sum(1 for i in range(total) if classification_lines[i] == output_lines[i])
    accuracy = (correct / total) * 100

    return accuracy


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python runnet0.py <Learning File> <Test File>")
        sys.exit(1)

    network = NeuralNetwork(0, 0, 0)
    #argv[1]= 'wnet0'
    network.load_weights(sys.argv[1])
    classify_data(network, sys.argv[2], 'output0.txt')
    accuracy= calculate_accuracy('classification0.txt', 'output0.txt')
    print(f"Accuracy rate: {accuracy:.2f}%")
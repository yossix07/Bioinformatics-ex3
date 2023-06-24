import sys
import random
import numpy as np

INPUT_SIZE = 16
HIDDEN_LAYER_SIZE = 10
LABEL_SIZE = 1
GENERATIONS = 100
POPULATION_SIZE = 120
MUTATION_RATE = 0.3
MAX_MUTATION_RATE = 0.9
REPLACEMENT_RATE = 0.4
REPLACEMENT_SIZE = int(POPULATION_SIZE * REPLACEMENT_RATE)
EPSILON = 0.0001
SAME_FITNESS_THRESHOLD = 12
flag_best_save = 0
SCALE = 0.1

class NeuralNetwork:
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_LAYER_SIZE, output_size=LABEL_SIZE):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_weights = np.random.uniform(low=0, high=0.5, size=(input_size, hidden_size))
        self.output_weights = np.random.uniform(low=0, high=0.5, size=(hidden_size, output_size))
        self.hidden_bias = np.random.uniform(low=0, high=0.5, size=hidden_size)
        self.output_bias = np.random.uniform(low=0, high=0.5, size=output_size)

    def forward(self, inputs):
        hidden_layer = np.dot(inputs, self.hidden_weights) + self.hidden_bias
        hidden_layer_activation = self.sigmoid(hidden_layer)
        output_layer = np.dot(hidden_layer_activation, self.output_weights) + self.output_bias
        output = self.sigmoid(output_layer)
        if output > 0.5:
            y = 1
        else:
            y = 0
        return y

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def relu(self, x):
        return np.maximum(0, x)


class GeneticAlgorithm:
    def __init__(self, test_data, learning_data, population_size=POPULATION_SIZE, mutation_rate=MUTATION_RATE):
        self.learning_data = learning_data
        self.learning_data_size = len(learning_data)
        self.test_data = test_data
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.network = NeuralNetwork()

    def generate_population(self):
        population = []
        for _ in range(self.population_size):
            weights = np.random.uniform(low=-1, high=1, size=self.network.hidden_weights.size + self.network.output_weights.size + 2)
            population.append(weights)
        return population

    def decode_weights(self, weights):
        self.hidden_bias = weights[-2]
        self.output_bias = weights[-1]
        weights = weights[:-2]
        hidden_weights = np.reshape(weights[:self.network.hidden_weights.size], self.network.hidden_weights.shape)
        output_weights = np.reshape(weights[self.network.hidden_weights.size:], self.network.output_weights.shape)
        self.network.hidden_weights = hidden_weights
        self.network.output_weights = output_weights

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2

    def mutate(self, weights):
        weights = np.copy(weights)
        for i in range(len(weights)):
            if random.random() < self.mutation_rate:
                mutation = np.random.normal(loc=-0.1, scale=SCALE)
                weights[i] += mutation
                weights[i] = np.clip(weights[i], -1, 1)  # Ensure the weights stay within the desired range

        # Separate the bias values from the weights array
        hidden_bias = weights[-2]
        output_bias = weights[-1]

        # Mutate the bias values
        if random.random() < self.mutation_rate:
            hidden_bias += np.random.normal(loc=-0.1, scale=SCALE)
            hidden_bias = np.clip(hidden_bias, -1, 1)

        if random.random() < self.mutation_rate:
            output_bias += np.random.normal(loc=-0.1, scale=SCALE)
            output_bias = np.clip(output_bias, -1, 1)

        # Update the bias values in the weights array
        weights[-2] = hidden_bias
        weights[-1] = output_bias

        return weights
    
    def select_parents(self, population, fitness_scores, tournament_size=5):
        parents = []

        # Select 2 parents
        for _ in range(2):
            tournament_candidates = random.sample(range(POPULATION_SIZE), tournament_size)
            tournament_scores = [fitness_scores[i] for i in tournament_candidates]
            winner_index = tournament_candidates[tournament_scores.index(max(tournament_scores))]
            parents.append(population[winner_index])

        return parents


    def calculate_fitness(self, weights):
        genetic_algorithm.decode_weights(weights)
        correct_predictions=0
        for data in self.learning_data:
            inputs = np.array(data[:-1], dtype=float)
            expected_output = np.array(data[-1], dtype=float)
            output = self.network.forward(inputs)
            if output == expected_output:
                correct_predictions += 1
        return correct_predictions / self.learning_data_size


    def replace_population(self, population, offspring, fitness_scores):
        worst_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[:3*REPLACEMENT_SIZE]

        # Find indices of individuals with best fitness scores from the offspring
        best_offspring_indices = sorted(range(len(offspring)), key=lambda i: self.calculate_fitness(offspring[i]),
                                        reverse=True)[:REPLACEMENT_SIZE]

        for i in range(REPLACEMENT_SIZE):
            population[worst_indices[i]] = offspring[best_offspring_indices[i]]

        return population

    def run(self):
        population = self.generate_population()
        fittest_weights = []
        best_fitness = 0
        prev_best_fitness = 0
        same_fitness_count = 0

        for generation in range(GENERATIONS):
            fitness_scores = []
            offspring = []

            for weights in population:
                fitness = self.calculate_fitness(weights)
                fitness_scores.append(fitness)

            # Crossover - Perform crossover to create new offspring
            for _ in range(POPULATION_SIZE//2):
                parent1, parent2 = self.select_parents(population, fitness_scores)
                child1, child2 = self.crossover(parent1, parent2)
                offspring.append(self.mutate(child1))
                offspring.append(self.mutate(child2))

            # Replace the weakest weights in the population with the mutated offspring
            population = self.replace_population(population, offspring, fitness_scores)
            fitness_scores = []
            mutated_population = []
            for weights in population:
                fitness = self.calculate_fitness(weights)
                fitness_scores.append(fitness)
                mutated_population.append(self.mutate(weights))
            
            population = self.replace_population(population, mutated_population, fitness_scores)

            # Update the weights with the fittest weights from the population
            fittest_weights_index = np.argmax(fitness_scores)
            fittest_weights = population[fittest_weights_index]
            self.decode_weights(fittest_weights)
            best_fitness = fitness_scores[fittest_weights_index]

            print("Generation:", generation, "Best Fitness:", best_fitness)

            if best_fitness >= 0.99:
                return fittest_weights
            if best_fitness == prev_best_fitness:
                same_fitness_count += 1
            else:
                same_fitness_count = 0
                self.mutation_rate = MUTATION_RATE
            prev_best_fitness = best_fitness
            if same_fitness_count > SAME_FITNESS_THRESHOLD:
                # return fittest_weights
                self.mutation_rate = MAX_MUTATION_RATE

        return fittest_weights


def save_weights_to_file(file_path, input_size, hidden_size, output_size, hidden_weights, output_weights):
    with open(file_path, 'w') as file:
        # Write input size, hidden size, and output size on the first line
        file.write(str(input_size) + ' ' + str(hidden_size) + ' ' + str(output_size) + '\n')

        # Write hidden weights as a list on the second line
        hidden_weights_str = ', '.join([str(weight) for weight in hidden_weights.flatten()])
        file.write('[' + hidden_weights_str + ']\n')

        # Write output weights as a list on the third line
        output_weights_str = ', '.join([str(weight) for weight in output_weights.flatten()])
        file.write('[' + output_weights_str + ']\n')


def parse_files():
    learning_file = open(sys.argv[1], 'r')
    learning_data = parse_file(learning_file)
    learning_file.close()

    test_file = open(sys.argv[2], 'r')
    test_data = parse_file(test_file)
    test_file.close()

    return learning_data, test_data

def parse_file(file):
    file_data = []
    for line in file:
        if line == '\n':
            continue
        input, label = line.split()
        np_line = np.array([int(ch) for ch in input if ch.isdigit()], dtype=int)
        np_line = np.append(np_line, int(label))
        file_data.append(np_line)
    return file_data

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python buildnet.py <Learning File> <Test File>")
        sys.exit(1)

    learning_data, test_data = parse_files()
    genetic_algorithm = GeneticAlgorithm(test_data, learning_data)
    best_weights = genetic_algorithm.run()
    #save in file
    save_weights_to_file("wnet0", INPUT_SIZE, HIDDEN_LAYER_SIZE, LABEL_SIZE, best_weights[:HIDDEN_LAYER_SIZE],
                                 best_weights[HIDDEN_LAYER_SIZE:])

  #  After the genetic algorithm steps, you can use the trained network to make predictions on the test data
    correct=0
    size = len(test_data)
    for data in test_data:
        inputs = np.array(data[:-1], dtype=float)
        label = np.array(data[-1], dtype=float)
        output = genetic_algorithm.network.forward(inputs)
        if label == output:
            correct += 1
    print('Accuracy: ', correct / size)
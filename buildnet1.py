import sys
import random
import numpy as np

INPUT_SIZE = 16
HIDDEN_LAYER_SIZE = 10
LABEL_SIZE = 1
GENERATIONS = 100
POPULATION_SIZE = 160
MUTATION_RATE = 0.4
MAX_MUTATION_RATE = 0.9
REPLACEMENT_RATE = 0.4
REPLACEMENT_SIZE = int(POPULATION_SIZE * REPLACEMENT_RATE)
EPSILON = 0.0001
SAME_FITNESS_THRESHOLD = 12
SAVE_TO_FILE = True
SCALE = 0.3

class NeuralNetwork:
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_LAYER_SIZE, output_size=LABEL_SIZE):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_weights = np.random.uniform(low=-1, high=1, size=(input_size, hidden_size))
        self.output_weights = np.random.uniform(low=-1, high=1, size=(hidden_size, output_size))

    # forward propagation
    def forward(self, inputs):
        hidden_layer = np.dot(inputs, self.hidden_weights)
        hidden_layer_activation = self.sigmoid(hidden_layer)
        output_layer = np.dot(hidden_layer_activation, self.output_weights)
        output = self.sigmoid(output_layer)
        if output >= 0.5:
            y = 1
        else:
            y = 0
        return y

    # activation functions
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

    # create a random population of weights
    def generate_population(self):
        population = []
        for _ in range(self.population_size):
            weights = np.random.uniform(low=-0.5, high=0.5,
                                        size=self.network.hidden_weights.size + self.network.output_weights.size)
            population.append(weights)
        return population

    # decode the weights array into the network's weights and biases
    def decode_weights(self, weights):
        hidden_weights = np.reshape(weights[:self.network.hidden_weights.size], self.network.hidden_weights.shape)
        output_weights = np.reshape(weights[self.network.hidden_weights.size:], self.network.output_weights.shape)
        self.network.hidden_weights = hidden_weights
        self.network.output_weights = output_weights

    # crossover two parents to produce two offspring
    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2

    # mutate the weights array
    def mutate(self, weights):
        weights = np.copy(weights)
        for i in range(len(weights)):
            if random.random() < self.mutation_rate:
                mutation = np.random.normal(loc=-0.1, scale=SCALE)
                weights[i] += mutation
                weights[i] = np.clip(weights[i], -1, 1)
        return weights

    # select parents using tournament selection
    def select_parents(self, population, fitness_scores, tournament_size=5):
        parents = []

        # Select 2 parents
        for _ in range(2):
            tournament_candidates = random.sample(range(POPULATION_SIZE), tournament_size)
            tournament_scores = [fitness_scores[i] for i in tournament_candidates]
            winner_index = tournament_candidates[tournament_scores.index(max(tournament_scores))]
            parents.append(population[winner_index])

        return parents

    # calculate the fitness score of an individual
    def calculate_fitness(self, weights):
        self.decode_weights(weights)
        correct_predictions = 0
        for data in self.learning_data:
            inputs = np.array(data[:-1], dtype=float)
            expected_output = np.array(data[-1], dtype=float)
            output = self.network.forward(inputs)
            if output == expected_output:
                correct_predictions += 1
        return correct_predictions / self.learning_data_size

    # replace the worst individuals in the population with the best offspring
    def replace_population(self, population, offspring, fitness_scores):
        worst_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[:3 * REPLACEMENT_SIZE]

        # Find indices of individuals with best fitness scores from the offspring
        best_offspring_indices = sorted(range(len(offspring)), key=lambda i: self.calculate_fitness(offspring[i]),
                                        reverse=True)[:REPLACEMENT_SIZE]

        for i in range(REPLACEMENT_SIZE):
            population[worst_indices[i]] = offspring[best_offspring_indices[i]]

        return population

    # run the genetic algorithm
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
            for _ in range(POPULATION_SIZE // 2):
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

            if best_fitness == prev_best_fitness:
                same_fitness_count += 1
            else:
                same_fitness_count = 0
                self.mutation_rate = MUTATION_RATE
            prev_best_fitness = best_fitness
            if same_fitness_count > SAME_FITNESS_THRESHOLD:
                self.mutation_rate = MAX_MUTATION_RATE

        return fittest_weights


# Save the weights to a file
def save_weights_to_file(file_path, input_size, hidden_size, output_size, weights):
    hidden_weights = weights[:hidden_size * input_size]
    output_weights = weights[hidden_size * input_size:]
    with open(file_path, 'w') as file:
        # Write input size, hidden size, and output size on the first line
        file.write(str(input_size) + ' ' + str(hidden_size) + ' ' + str(output_size) + '\n')

        # Write hidden weights as a list on the second line
        hidden_weights_str = ', '.join([str(weight) for weight in hidden_weights.flatten()])
        file.write('[' + hidden_weights_str + ']\n')

        # Write output weights as a list on the third line
        output_weights_str = ', '.join([str(weight) for weight in output_weights.flatten()])
        file.write('[' + output_weights_str + ']\n')


# Parse the learning and test files
def parse_files():
    learning_file = open(sys.argv[1], 'r')
    learning_data = parse_file(learning_file)
    learning_file.close()

    test_file = open(sys.argv[2], 'r')
    test_data = parse_file(test_file)
    test_file.close()

    return learning_data, test_data


# Parse a file and return a list of numpy arrays
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
    # save in file
    if SAVE_TO_FILE:
        save_weights_to_file("wnet1", INPUT_SIZE, HIDDEN_LAYER_SIZE, LABEL_SIZE, best_weights)
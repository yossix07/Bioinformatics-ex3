import sys
import random
import numpy as np

INPUT_SIZE = 16
HIDDEN_LAYER_SIZE = 10
LABEL_SIZE = 1
GENERATIONS = 1
POPULATION_SIZE = 100
MUTATION_RATE = 0.1
REPLACEMENT_RATE = 0.15
REPLACEMENT_SIZE = int(POPULATION_SIZE * REPLACEMENT_RATE)
EPSILON = 0.0001

class NeuralNetwork:
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_LAYER_SIZE, output_size=LABEL_SIZE):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_weights = np.random.uniform(low=-1, high=1, size=(input_size, hidden_size))
        self.output_weights = np.random.uniform(low=-1, high=1, size=(hidden_size, output_size))

    def forward(self, inputs):
        hidden_layer = np.dot(inputs, self.hidden_weights)
        hidden_layer_activation = self.sigmoid(hidden_layer)
        output_layer = np.dot(hidden_layer_activation, self.output_weights)
        output = self.sigmoid(output_layer)
        return output

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

class GeneticAlgorithm:
    def __init__(self, learning_data, population_size=POPULATION_SIZE, mutation_rate=MUTATION_RATE):
        self.learning_data = learning_data
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.network = NeuralNetwork()

    def generate_population(self):
        population = []
        for _ in range(self.population_size):
            weights = np.random.uniform(low=-1, high=1, size=self.network.hidden_weights.size + self.network.output_weights.size)
            population.append(weights)
        return population

    def decode_weights(self, weights):
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
        for i in range(len(weights)):
            if random.random() < self.mutation_rate:
                weights[i] += np.random.uniform(low=-0.5, high=0.5)
        return weights
    
    def select_parents(self, population, fitness_scores, tournament_size=5):
        parents = []
        
        # Select 2 parents
        for _ in range(2):
            tournament_candidates = random.sample(range(len(population)), tournament_size)
            tournament_scores = [fitness_scores[i] for i in tournament_candidates]
            winner_index = tournament_candidates[tournament_scores.index(max(tournament_scores))]
            parents.append(population[winner_index])
        
        return parents
    
    def calculate_fitness(self, weights):
        genetic_algorithm.decode_weights(weights)
        total_error = 0.0
        for data in self.learning_data:
            inputs = np.array(data[:-1], dtype=float)
            expected_output = np.array(data[-1], dtype=float)
            output = self.network.forward(inputs)
            total_error += np.abs(expected_output - output)
        return 1.0 / (total_error + 1) # Inverse of the error as fitness score
    

    def replace_population(self, population, offspring, fitness_scores):
        worst_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[:3*REPLACEMENT_SIZE]

        # Find indices of individuals with best fitness scores from the existing population
        best_population_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:REPLACEMENT_SIZE]

        # Find indices of individuals with best fitness scores from the offspring
        best_offspring_indices = sorted(range(len(offspring)), key=lambda i: self.calculate_fitness(offspring[i]),
                                        reverse=True)[:REPLACEMENT_SIZE]

        # Replace worst fitness scores with best fitness scores from the existing population
        for i in range(REPLACEMENT_SIZE):
            population[worst_indices[i]] = population[best_population_indices[0]]
            population[worst_indices[i + REPLACEMENT_SIZE]] = population[best_population_indices[i]]
            population[worst_indices[i + 2 * REPLACEMENT_SIZE]] = offspring[best_offspring_indices[i]]

        return population
    
    def run(self):
        population = self.generate_population()

        for generation in range(GENERATIONS):
            # Evaluation - Calculate fitness for each weights in the population
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

            # Mutation - Apply mutation to the offspring
            # mutated_offspring = [self.mutate(weights) for weights in offspring]

            # Replace the weakest weightss in the population with the mutated offspring
            population = self.replace_population(population, offspring, fitness_scores)

            # Update the weights with the fittest weights from the population
            fittest_weights = population[np.argmax(fitness_scores)]
            self.decode_weights(fittest_weights)
            # Print the fitness score of the fittest weights
            print("Generation:", generation, "Best Fitness:", fitness_scores[np.argmax(fitness_scores)])

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
    genetic_algorithm = GeneticAlgorithm(learning_data)
    genetic_algorithm.run()

    # After the genetic algorithm steps, you can use the trained network to make predictions on the test data
    # for data in test_data:
    #     inputs = np.array(data[:-1], dtype=float)
    #     label = np.array(data[-1], dtype=float)
    #     inputs = np.array(input, dtype=float)
    #     output = genetic_algorithm.network.forward(inputs)
    #     print("Input:", inputs, "Label:", label, "Prediction:", output, "Error Distance:", np.abs(label - output))
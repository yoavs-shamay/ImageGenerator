from neuralnetwork import NeuralNetwork
import random
import numpy as np


class GeneratorNetwork:
    def __init__(self, generator_layers, generator_activations, generator_activation_derivatives, discriminator_layers,
                 discriminator_activations, discriminator_activation_derivatives):
        self.generator = NeuralNetwork(generator_layers, generator_activations, generator_activation_derivatives)
        self.discriminator = NeuralNetwork(discriminator_layers, discriminator_activations,
                                           discriminator_activation_derivatives)

    def feedforward(self, a):
        return self.generator.feedforward(a)

    def get_result(self, a):
        return self.generator.get_result(a)

    def generate(self):
        rand = np.random.randn(self.generator.layers[0])
        return self.generator.get_result(rand)

    def train(self, data, iteration_count, learning_rate, batch_size, cost_derivative):
        for iteration in range(iteration_count):
            random.shuffle(data)
            i = 0
            while i < len(data):
                batch_x = data[i:min(len(data), i + batch_size)]
                batch_y = [[1] for j in range(len(batch_x))]
                self.discriminator.train(batch_x, batch_y, 1, learning_rate, batch_size, cost_derivative)
                batch_y = [[0] for j in range(len(batch_x))]
                generate_inputs = np.array([np.random.randn(self.generator.layers[0]) for j in range(len(batch_x))])
                generated = [self.generator.get_result(generate_inputs[j]) for j in range(len(batch_x))]
                self.discriminator.train(generated, batch_y, 1, learning_rate, batch_size, cost_derivative)
                weights_deltas = [np.zeros(self.generator.weights[i].shape).astype(float) for i in
                                  range(len(self.generator.weights))]
                biases_deltas = [np.zeros(self.generator.biases[i].shape).astype(float) for i in
                                 range(len(self.generator.biases))]
                for value in range(len(batch_x)):
                    discriminator_end = self.discriminator.get_current_change(generated[value], [1], cost_derivative,
                                                                              self.discriminator.activation_derivatives,
                                                                              self.generator.activation_derivatives[-1])
                    self.generator.backpropagation(generate_inputs[value], [1], cost_derivative,
                                                   self.generator.activation_derivatives, weights_deltas, biases_deltas,
                                                   discriminator_end)
                for i in range(len(self.generator.weights)):
                    self.generator.weights[i] -= learning_rate * weights_deltas[i] / batch_size
                    self.generator.biases[i] -= learning_rate * biases_deltas[i] / batch_size
                i += batch_size

    def export(self):
        generator_model = self.generator.export()
        discriminator_model = self.discriminator.export()
        return generator_model + '\n' + discriminator_model

    def import_model(self, model_string):
        lines = model_string.split('\n')
        self.generator.import_model(lines[0])
        self.discriminator.import_model(lines[1])

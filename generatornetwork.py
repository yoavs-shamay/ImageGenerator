from neuralnetwork import NeuralNetwork
import random
import cupy as np


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

    def train(self, data, iteration_count, learning_rate, batch_size, cost_derivative, debug=False):
        for iteration in range(iteration_count):
            random.shuffle(data)
            i = 0
            while i < len(data):
                batch_x = data[i:min(len(data), i + batch_size)]
                batch_y = np.array([[1] for j in range(len(batch_x))]).astype(float)
                noise = np.random.uniform(low=0,high=0.2,size=(len(batch_y), 1))
                batch_y -= noise
                self.discriminator.train(batch_x, batch_y, 1, learning_rate, batch_size, cost_derivative)
                batch_y = np.array([[0] for j in range(len(batch_x))]).astype(float)
                noise = np.random.uniform(low=0,high=0.2,size=(len(batch_y), 1))
                batch_y += noise
                generate_inputs = np.array([np.random.randn(self.generator.layers[0]) for j in range(len(batch_x))])
                generated = [self.generator.get_result(generate_inputs[j]) for j in range(len(batch_x))]
                self.discriminator.train(generated, batch_y, 1, learning_rate, batch_size, cost_derivative)
                weights_deltas = [np.zeros(self.generator.weights[j].shape).astype(float) for j in
                                  range(len(self.generator.weights))]
                biases_deltas = [np.zeros(self.generator.biases[j].shape).astype(float) for j in
                                 range(len(self.generator.biases))]
                for value in range(len(batch_x)):
                    discriminator_end = self.discriminator.get_current_change(generated[value], [1], cost_derivative,
                                                                              self.discriminator.activation_derivatives,
                                                                              self.generator.activation_derivatives[-1])
                    self.generator.backpropagation(generate_inputs[value], [1], cost_derivative,
                                                   self.generator.activation_derivatives, weights_deltas, biases_deltas,
                                                   discriminator_end)
                for j in range(len(self.generator.weights)):
                    self.generator.weights[j] -= learning_rate * weights_deltas[j] / batch_size
                    self.generator.biases[j] -= learning_rate * biases_deltas[j] / batch_size
                i += batch_size
                if debug:
                    print(i,'/',len(data))
                    count = 0
                    for _ in range(10):
                        cur = self.generate()
                        dis = self.discriminator.get_result(cur)
                        if dis[0] > 0.5:
                            count += 1
                    print(count, '/', 10)
                    if i % 1000 == 0:
                        text = self.export()
                        file = open('mnist_generator.json', 'w')
                        file.write(text)
                        file.close()

    def export(self):
        generator_model = self.generator.export()
        discriminator_model = self.discriminator.export()
        return generator_model + '\n' + discriminator_model

    def import_model(self, model_string):
        lines = model_string.split('\n')
        self.generator.import_model(lines[0])
        self.discriminator.import_model(lines[1])

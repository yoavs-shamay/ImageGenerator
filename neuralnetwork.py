import random
import numpy as np
import json


class NeuralNetwork:
    def __init__(self, layers, activations, activation_derivatives):
        self.layers = layers
        self.weights = []
        self.biases = []
        self.initialize_weights()
        self.activations = activations
        self.activation_derivatives = activation_derivatives

    def initialize_weights(self):
        for i in range(len(self.layers) - 1):
            self.weights.append(np.random.randn(self.layers[i + 1], self.layers[i]).astype(float))
            self.biases.append(np.zeros(self.layers[i + 1]).astype(float))

    def get_result(self, a):
        i = 0
        for i in range(len(self.weights)):
            new_a = []
            cur = np.dot(self.weights[i], a) + self.biases[i]
            for j in range(len(self.weights[i])):
                new_a.append(self.activations[i](cur[j]))
            a = np.array(new_a)
        return new_a

    def feedforward(self, a):
        res = [a]
        i = 0
        for i in range(len(self.weights)):
            new_a = []
            cur = np.dot(self.weights[i], a) + self.biases[i]
            for j in range(len(self.weights[i])):
                new_a.append(self.activations[i](cur[j]))
            res.append(new_a)
            a = np.array(new_a)
        return res

    def train(self, data_x, data_y, iteration_count, learning_rate, batch_size, cost_derivative):
        for iteration in range(iteration_count):
            zipped = list(zip(data_x, data_y))
            random.shuffle(zipped)
            data_x, data_y = zip(*zipped)
            i = 0
            while i < len(data_x):
                batch_x = data_x[i:min(len(data_x), i + batch_size)]
                batch_y = data_y[i:min(len(data_x), i + batch_size)]
                j = 0
                weights_deltas = [np.zeros(self.weights[i].shape).astype(float) for i in range(len(self.weights))]
                biases_deltas = [np.zeros(self.biases[i].shape).astype(float) for i in range(len(self.biases))]
                for x, y in zip(batch_x, batch_y):
                    self.backpropagation(x, y, cost_derivative, self.activation_derivatives, weights_deltas,
                                         biases_deltas)
                for n in range(len(self.weights)):
                    self.weights[n] -= learning_rate * weights_deltas[n]
                    self.biases[n] -= learning_rate * biases_deltas[n]
                i += batch_size
                if i % 10000 == 0:
                    print(i, '/', len(data_x))

    def backpropagation(self, x, y, cost_derivative, activation_derivatives, weights_deltas, biases_deltas,
                        current_change=None):
        values = self.feedforward(x)
        if current_change is None:
            derivative_cost = cost_derivative(values[-1], y)
            current_change = activation_derivatives[-1](values[-1]) * derivative_cost
        for i in range(len(self.weights) - 1, -1, -1):
            current_weights_deltas = np.outer(current_change, values[i])
            current_biases_deltas = np.array(current_change)
            weights_deltas[i] += current_weights_deltas
            biases_deltas[i] += current_biases_deltas
            if i > 0:
                current_change = np.dot(self.weights[i].transpose(), current_change) * activation_derivatives[i - 1](
                    values[i])

    def get_current_change(self, x, y, cost_derivative, activation_derivatives, last_activation_derivative,
                           current_change=None):
        values = self.feedforward(x)
        if current_change is None:
            derivative_cost = cost_derivative(values[-1], y)
            current_change = activation_derivatives[-1](values[-1]) * derivative_cost
        new_activation_derivatives = [last_activation_derivative] + activation_derivatives
        for i in range(len(self.weights) - 1, -1, -1):
            current_change = np.dot(self.weights[i].transpose(), current_change) * new_activation_derivatives[i](
                values[i])
        return current_change

    def test_classification(self, data_x, data_y):
        res = 0
        for x, y in zip(data_x, data_y):
            if np.argmax(self.get_result(x)) == np.argmax(y):
                res += 1
        return res / len(data_x)

    def test_regression(self, data_x, data_y):
        res = 0
        for x, y in zip(data_x, data_y):
            res += (self.get_result(x)[0] - y[0]) ** 2
        return res / len(data_x)

    def export(self):
        weights_list = []
        biases_list = []
        for i in range(len(self.weights)):
            weights_list.append(self.weights[i].tolist())
            biases_list.append(self.biases[i].tolist())
        model = {'weights': weights_list, 'biases': biases_list}
        return json.dumps(model)

    def import_model(self, model_str):
        data = json.loads(model_str)
        self.weights = [np.array(data['weights'][i]) for i in range(len(data['weights']))]
        self.biases = [np.array(data['biases'][i]) for i in range(len(data['biases']))]

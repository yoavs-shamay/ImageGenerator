from neuralnetwork import NeuralNetwork
from functions import *
from MNIST import *
import time

LAYERS = [784, 30, 10]
ACTIVATIONS = [sigmoid, sigmoid]
activations_derivatives = [derivative[ACTIVATIONS[i]] for i in range(1, len(ACTIVATIONS))]
COST = lms
cost_derivative = derivative[COST]
ITERATION_COUNT = 1
LEARNING_RATE = 5
BATCH_SIZE = 10

train_x, train_y = get_train()
test_x, test_y = get_test()

start = time.time()
model = NeuralNetwork(LAYERS, ACTIVATIONS, activations_derivatives)
model.train(train_x, train_y, ITERATION_COUNT, LEARNING_RATE, BATCH_SIZE, cost_derivative)
print(model.test_classification(test_x, test_y) * 100, '%', sep='')
text = model.export()
with open('mnist.json', 'w') as file:
    file.write(text)

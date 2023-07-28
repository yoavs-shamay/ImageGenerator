from generatornetwork import GeneratorNetwork
import numpy as np
from functions import *
from load_cifar import load_data

GENERATOR_LAYERS = [10, 10, 3072]
GENERATOR_ACTIVATIONS = [leaky_relu, sigmoid]
generator_activations_derivatives = [derivative[GENERATOR_ACTIVATIONS[i]] for i in range(1, len(GENERATOR_ACTIVATIONS))]
DISCRIMINATOR_LAYERS = [3072, 10, 1]
DISCRIMINATOR_ACTIVATIONS = [leaky_relu, sigmoid]
discriminator_activations_derivatives = [derivative[DISCRIMINATOR_ACTIVATIONS[i]] for i in
                                         range(1, len(DISCRIMINATOR_ACTIVATIONS))]
COST = lms
cost_derivative = derivative[COST]
ITERATION_COUNT = 1
LEARNING_RATE = 5
BATCH_SIZE = 10

model = GeneratorNetwork(GENERATOR_LAYERS, GENERATOR_ACTIVATIONS, generator_activations_derivatives,
                         DISCRIMINATOR_LAYERS, DISCRIMINATOR_ACTIVATIONS, discriminator_activations_derivatives)
data = load_data()
model.train(data, ITERATION_COUNT, LEARNING_RATE, BATCH_SIZE, cost_derivative)
text = model.export()
file = open('cifar.json', 'w')
file.write(text)
file.close()

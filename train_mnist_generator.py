from generatornetwork import GeneratorNetwork
from functions import *
from MNIST import *

GENERATOR_LAYERS = [20, 16, 16, 784]
GENERATOR_ACTIVATIONS = [leaky_relu, leaky_relu, tanh]
generator_activations_derivatives = [derivative[GENERATOR_ACTIVATIONS[i]] for i in range(1, len(GENERATOR_ACTIVATIONS))]
DISCRIMINATOR_LAYERS = [784, 16, 16, 1]
DISCRIMINATOR_ACTIVATIONS = [leaky_relu, leaky_relu, sigmoid]
discriminator_activations_derivatives = [derivative[DISCRIMINATOR_ACTIVATIONS[i]] for i in
                                         range(1, len(DISCRIMINATOR_ACTIVATIONS))]
COST = lms
cost_derivative = derivative[COST]
ITERATION_COUNT = 500
LEARNING_RATE = 0.5
BATCH_SIZE = 5

model = GeneratorNetwork(GENERATOR_LAYERS, GENERATOR_ACTIVATIONS, generator_activations_derivatives,
                         DISCRIMINATOR_LAYERS, DISCRIMINATOR_ACTIVATIONS, discriminator_activations_derivatives)

file = open('mnist_generator.json', 'r')
text = file.read()
file.close()
model.import_model(text)

data = np.array(get_train()[0].tolist() + get_test()[0].tolist())
model.train(data, ITERATION_COUNT, LEARNING_RATE, BATCH_SIZE, cost_derivative, debug=True)
text = model.export()
file = open('mnist_generator.json', 'w')
file.write(text)
file.close()

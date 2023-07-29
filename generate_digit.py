from generatornetwork import GeneratorNetwork
from functions import *
from PIL import Image
import cupy as np

GENERATOR_LAYERS = [20, 16, 16, 784]
GENERATOR_ACTIVATIONS = [leaky_relu, leaky_relu, tanh]
generator_activations_derivatives = [derivative[GENERATOR_ACTIVATIONS[i]] for i in range(1, len(GENERATOR_ACTIVATIONS))]
DISCRIMINATOR_LAYERS = [784, 16, 16, 1]
DISCRIMINATOR_ACTIVATIONS = [leaky_relu, leaky_relu, sigmoid]
discriminator_activations_derivatives = [derivative[DISCRIMINATOR_ACTIVATIONS[i]] for i in
                                         range(1, len(DISCRIMINATOR_ACTIVATIONS))]

model = GeneratorNetwork(GENERATOR_LAYERS, GENERATOR_ACTIVATIONS, generator_activations_derivatives,
                         DISCRIMINATOR_LAYERS, DISCRIMINATOR_ACTIVATIONS, discriminator_activations_derivatives)

file = open('mnist_generator.json', 'r')
text = file.read()
file.close()
model.import_model(text)
image_data = model.generate()
image_pixels = []
for i in range(28):
    image_pixels.append([])
    for j in range(28):
        image_pixels[i].append(image_data[i * 28 + j])
image = Image.fromarray(np.array(image_pixels).get(), 'L')
image.show()

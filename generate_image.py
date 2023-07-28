from generatornetwork import GeneratorNetwork
import numpy as np
from functions import *
from PIL import Image

GENERATOR_LAYERS = [10, 10, 3072]
GENERATOR_ACTIVATIONS = [leaky_relu, sigmoid]
generator_activations_derivatives = [derivative[GENERATOR_ACTIVATIONS[i]] for i in range(1, len(GENERATOR_ACTIVATIONS))]
DISCRIMINATOR_LAYERS = [3072, 10, 1]
DISCRIMINATOR_ACTIVATIONS = [leaky_relu, sigmoid]
discriminator_activations_derivatives = [derivative[DISCRIMINATOR_ACTIVATIONS[i]] for i in
                                         range(1, len(DISCRIMINATOR_ACTIVATIONS))]
COST = lms
cost_derivative = derivative[COST]
ITERATION_COUNT = 500
LEARNING_RATE = 5
BATCH_SIZE = 10

model = GeneratorNetwork(GENERATOR_LAYERS, GENERATOR_ACTIVATIONS, generator_activations_derivatives,
                         DISCRIMINATOR_LAYERS, DISCRIMINATOR_ACTIVATIONS, discriminator_activations_derivatives)

file = open('cifar.json', 'r')
text = file.read()
file.close()
model.import_model(text)

image_data = model.generate()
image_pixels = []
for i in range(32):
    image_pixels.append([])
    for j in range(32):
        image_pixels[i].append([image_data[i * 32 + j] * 255, image_data[i * 32 + j + 1024] * 255, image_data[i * 32 + j + 2048] * 255])
image = Image.fromarray(np.array(image_pixels, dtype=np.uint8))
image.show()

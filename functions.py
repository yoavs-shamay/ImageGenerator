import numpy as np


def sigmoid_(x):
    return 1.0 / (1.0 + np.exp(-x))


sigmoid = np.vectorize(sigmoid_)


def sigmoid_derivative_(x):
    return sigmoid(x) * (1 - sigmoid(x))


sigmoid_derivative = np.vectorize(sigmoid_derivative_)


def relu_(x):
    return np.maximum(x, 0)


relu = np.vectorize(relu_)


def relu_derivative_(x):
    return 1 * (x > 0)


relu_derivative = np.vectorize(relu_derivative_)


def leaky_relu_(x):
    return np.maximum(x, 0.01 * x)


leaky_relu = np.vectorize(leaky_relu_)


def leaky_relu_derivative_(x):
    return 1 * (x > 0) + 0.01 * (x < 0)


leaky_relu_derivative = np.vectorize(leaky_relu_derivative_)


def lms(x, y):
    res = 0
    for i in range(len(x)):
        res += (x[i] - y[i]) ** 2
    return res / len(x)


def lms_derivative(x, y):
    res = []
    for i in range(len(x)):
        res.append(2 * (x[i] - y[i]))
    return res

def binary_crossentropy(x, y):
    res = 0
    for i in range(len(x)):
        res += y[i] * np.log(x[i]) + (1 - y[i]) * np.log(1 - x[i])
    return -res / len(x)


def binary_crossentropy_derivative(x, y):
    res = []
    for i in range(len(x)):
        res.append((x[i] - y[i]) / (x[i] * (1 - x[i])))
    return res


def tanh_(x):
    return np.tanh(x) / 2 + 0.5


tanh = np.vectorize(tanh_)


def tanh_derivative_(x):
    return (1 - tanh(x) ** 2) / 2


tanh_derivative = np.vectorize(tanh_derivative_)


derivative = {sigmoid: sigmoid_derivative, relu: relu_derivative, leaky_relu: leaky_relu_derivative, lms: lms_derivative, binary_crossentropy: binary_crossentropy_derivative, tanh: tanh_derivative}

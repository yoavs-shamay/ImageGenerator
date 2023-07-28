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


derivative = {sigmoid: sigmoid_derivative, relu: relu_derivative, leaky_relu: leaky_relu_derivative, lms: lms_derivative}

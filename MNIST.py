from keras.datasets import mnist
import cupy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((len(x_train), 784))
x_test = x_test.reshape((len(x_test), 784))
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = np.array([[1 if i == y_train[j] else 0 for i in range(10)] for j in range(len(y_train))])
y_test = np.array([[1 if i == y_test[j] else 0 for i in range(10)] for j in range(len(y_test))])


def get_train():
    return x_train, y_train


def get_test():
    return x_test, y_test

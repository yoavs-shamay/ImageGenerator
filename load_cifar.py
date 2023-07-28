import pickle
import numpy as np


def load_data():
    data = []
    for batch in ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']:
        with open('cifar-10/' + batch, 'rb') as f:
            data += pickle.load(f, encoding='bytes')[b'data'].tolist()
    data = np.array(data).astype(float)
    data /= 255
    return data

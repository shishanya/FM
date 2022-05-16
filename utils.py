from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np


def load_data():
    boston = datasets.load_boston()
    x = boston["data"]
    y = boston["target"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    y_train = y_train.reshape(len(y_train), 1)
    train_dataset = np.hstack((x_train, y_train))
    y_test = y_test.reshape(len(y_test), 1)
    test_dataset = np.hstack((x_test, y_test))
    return train_dataset, test_dataset


load_data()

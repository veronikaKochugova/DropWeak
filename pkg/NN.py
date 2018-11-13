import numpy as np

global ITER_COUNT
global DROP_WEAK_LIMIT
ITER_COUNT = 10000
DROP_WEAK_LIMIT = 0.0001


class NN:
    """class of neural network"""

    def __init__(self, i_count, o_count):
        """
            i_count - count of input nodes
            o_count - count of output nodes
        """
        np.random.seed(1)
        self.weights = 2 * np.random.random((i_count, o_count)) - 1

    def train(self, X, y, with_regulation=False):
        if with_regulation:
            self.weights = with_drop_weak(self.weights, X, y, DROP_WEAK_LIMIT)
        self.weights = without_regulation(self.weights, X, y)

    def predict(self, x):
        return sigmoid(np.dot(x, self.weights))


# Сигмоида
def sigmoid(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def without_regulation(w, X, y):
    for iter in range(ITER_COUNT):
        # прямое распространение
        l0 = X
        l1 = sigmoid(np.dot(l0, w))

        # насколько мы ошиблись?
        l1_error = y - l1

        # перемножим это с наклоном сигмоиды
        # на основе значений в l1
        l1_delta = l1_error * sigmoid(l1, True)  # !!!

        # обновим веса
        w += np.dot(l0.T, l1_delta)  # !!!
    return w


def drop_weak(x, limit):
    g = lambda x: np.abs(x) >= limit
    return x * g(x)


def with_drop_weak(w, X, y, limit):
    for iter in range(ITER_COUNT):
        # прямое распространение
        l0 = X
        l1 = sigmoid(np.dot(l0, w))
        # насколько мы ошиблись?
        l1_error = y - l1
        # перемножим это с наклоном сигмоиды
        # на основе значений в l1
        l1_delta = l1_error * sigmoid(l1, True)  # !!!
        # обновим веса
        w += np.dot(l0.T, l1_delta)  # !!!
        w = drop_weak(w, limit)
    return w

import math
import random

import numpy as np

A = "Iris-setosa"
B = "Iris-virginica"

color = {
    A: 'r',
    B: 'b'
}

value = {
    A: 1,
    B: 0
}

actual = {
    0: A,
    1: B
}


def get_vector_data(data_set):
    # matrix of features
    x = np.array([sample.height for sample in data_set])

    # matrix of values
    y = np.array([d.weight for d in data_set])
    return x, y


def get_normal_data_from_vector(x_vector, y_vector):
    X = np.array([[1, _x] for _x in x_vector])
    xt = X.transpose()

    y = [0] + y_vector
    y = y.transpose()

    return X, xt, y


def gradient_descent(x_vector, y_vector, cost_function, alpha=0.1, ep=0.1 ** 12, beta=np.array([0, 0]), limit=1000):
    X, xt, y = get_normal_data_from_vector(x_vector, y_vector)

    m = len(y)
    last = np.math.inf
    beta_track = []

    while limit:
        limit -= 1
        hypothesis = X.dot(beta)
        loss = hypothesis - y
        _gradient = np.dot(xt, loss) / m
        beta = beta - alpha * _gradient
        cost = cost_function(beta, x_vector, y_vector)
        beta_track += [beta]
        if abs(cost - last) < ep:
            break
        last = cost

    return beta[0], beta[1], beta_track


def load(file):
    handle = open(file)

    data = handle.read()

    data = data.split('\n')

    _x = []
    _y = []
    _class = []

    max_f1 = - math.inf
    min_f1 = math.inf

    max_f2 = - math.inf
    min_f2 = math.inf

    for line in data:

        if len(line.split(',')) != 3:
            continue

        f1, f2, _cls = tuple(line.split(','))

        _f1 = float(f1)
        _f2 = float(f2)

        if _f1 > max_f1:
            max_f1 = _f1
        if _f1 < min_f1:
            min_f1 = _f1
        if _f2 > max_f2:
            max_f2 = _f2
        if _f2 < min_f2:
            min_f2 = _f2

        _x.append(_f1)
        _y.append(_f2)
        _class.append(value[_cls])

    # normalize the data
    for i in range(len(_x)):
        _x[i] = (_x[i] - min_f1) / (max_f1 - min_f1)

    for i in range(len(_y)):
        _y[i] = (_y[i] - min_f2) / (max_f2 - min_f2)

    handle.close()

    return _x, _y, _class


def make_data():
    x, y, labels = load("iris.data")

    # class A indexes
    class_A = [i for i in range(len(labels)) if labels[i] == value[A]]
    # class B indexes
    class_B = [i for i in range(len(labels)) if labels[i] == value[B]]

    samples_A = random.sample(population=class_A, k=int(0.8 * len(class_A)))
    samples_B = random.sample(population=class_B, k=int(0.8 * len(class_B)))

    to = open('training_set.data', 'w')

    for i in samples_A:
        to.write('%s,%s,%s\n' % (x[i], y[i], actual[labels[i]]))

    for i in samples_B:
        to.write('%s,%s,%s\n' % (x[i], y[i], actual[labels[i]]))

    to.close()

    tto = open('test_set.data', 'w')

    for i in range(len(labels)):
        if i not in samples_A and i not in samples_B:
            tto.write('%s,%s,%s\n' % (x[i], y[i], actual[labels[i]]))

    tto.close()

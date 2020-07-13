import numpy as np


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

    while True:
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

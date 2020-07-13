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


def hypothesis(theta, X, n):
    h = np.ones((X.shape[0], 1))
    theta = theta.reshape(1, n + 1)
    for i in range(0, X.shape[0]):
        h[i] = float(np.matmul(theta, X[i]))
    h = h.reshape(X.shape[0])
    return h


def BGD(_theta, alpha, num_iters, h, X, y, n):
    cost = np.ones(num_iters)

    for i in range(0, num_iters):
        _theta[0] = _theta[0] - (alpha / X.shape[0]) * sum(h - y)
        for j in range(1, n + 1):
            _theta[j] = _theta[j] - (alpha / X.shape[0]) *
            sum((h - y) * X.transpose()[j])
    h = hypothesis(_theta, X, n)
    cost[i] = (1 / X.shape[0]) * 0.5 * sum(np.square(h - y))

    theta = theta.reshape(1, n + 1)
    return theta, cost


def gradient_descent(x_vector, y_vector, cost_function, alpha=0.1, ep=0.1 ** 12, beta=np.array([0, 0]), limit=1000):
    X, xt, y = get_normal_data_from_vector(x_vector, y_vector)

    m = len(y)
    last = np.math.inf
    beta_track = []

    while True:
        limit -= 1
        print(limit)
        hypothesis = X.dot(beta)
        loss = hypothesis - y
        _gradient = gradient(cost_function, beta, x_vector, y_vector)
        beta = beta - alpha * _gradient
        cost = cost_function(beta, x_vector, y_vector)

        beta_track += [beta]

        if abs(cost - last) < ep:
            break
        last = cost

    return beta[0], beta[1], beta_track

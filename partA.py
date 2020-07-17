from loader import load
import matplotlib.pyplot as plt
import numpy as np
from utils import gradient_descent, get_normal_data_from_vector, get_vector_data

ds1 = load("Dataset1.txt")
ds2 = load("Dataset2.txt")


def J(beta, x_vector, y_vector):
    X, xt, y = get_normal_data_from_vector(x_vector, y_vector)

    m = len(y)
    j = np.sum((X.dot(beta) - y) ** 2) / 2 / m

    return j


def closed_form(x_vector, y_vector):
    x, xt, y = get_normal_data_from_vector(x_vector, y_vector)

    s = np.matmul(xt, x)
    pseudo_inverse = np.linalg.inv(s)

    fs = np.matmul(pseudo_inverse, xt)

    # learned parameters
    theta = np.matmul(fs, y)

    return theta[0], theta[1]


def plot(t0, t1, data_set, label):
    line_points = [0, 1]
    line_values = [t0 + lp * t1 for lp in line_points]

    plt.plot([d.height for d in data_set], [d.weight for d in data_set], 'ko', markersize=3, label=label)

    plt.plot(line_points, line_values)

    plt.legend(loc='best')

    plt.ylabel('weight')
    plt.xlabel('height')

    plt.show()


def sigmoid(start, end):
    step_size = 0.01

    x = [start + i * step_size for i in range(0, int((end - start) // step_size))]
    y = []

    for _x in x:
        y += [1 / (1 + np.math.exp(- _x))]

    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":

    # # plot using gradient descent
    d = gradient_descent(*get_vector_data(ds1), J, limit=1e12)
    plot(d[0], d[1], ds1, "ds1 gradient descent")

    # plot using linear regression
    plot(*closed_form(*get_vector_data(ds1)), ds1, "ds1 closed form")

    # plot using gradient descent
    d2 = gradient_descent(*get_vector_data(ds2), J)
    plot(d2[0], d2[1], ds2, "ds2 gradient descent")

    # plot using linear regression
    plot(*closed_form(*get_vector_data(ds2)), ds2, "ds2 closed form")

    # plot the learned t1 and t2
    plt.plot(range(len(d[2])), [m[1] for m in d[2]], label='theta 1')
    plt.plot(range(len(d[2])), [m[0] for m in d[2]], label='theta 2')

    plt.legend(loc='best')

    plt.show()
    sigmoid(-10, 10)

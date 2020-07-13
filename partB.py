from math import ceil
from random import random
import numpy as np
from loader import load
import matplotlib.pyplot as plt
import partA as pa

from utils import get_vector_data, gradient_descent

ds2 = load("Dataset2.txt")

CLOSED_FORM = "closed_form"
GRADIENT_DESCENT = "gradient_descent"


def get_x_y(data_set):
    # matrix of features
    x = np.array([sample.height for sample in data_set])

    # matrix of values
    y = np.array([d.weight for d in data_set])

    return x, y


def kernel_function(xi, x0, tau=.005):
    return np.exp(- (xi - x0) ** 2 / (2 * tau))


def J(beta, x, y):
    return sum([
        np.exp(- (x[i] - x[j]) ** 2 / (2 * 0.005)) *
        (y[j] - (beta[0] + beta[1] * x[j]))
        for i in range(len(x))
        for j in range(len(x))
    ])


def closed_form(w, x, y):
    b = np.array([np.sum(w * y), np.sum(w * y * x)])
    A = np.array([[np.sum(w), np.sum(w * x)],
                  [np.sum(w * x), np.sum(w * x * x)]])
    theta = np.linalg.solve(A, b)

    return theta


def LWR(x, y, method="closed_form"):
    n = m = len(x)
    _learned = np.zeros(n)
    w = np.array([kernel_function(x, x[i]) for i in range(m)])

    if method == "closed_form":
        theta = closed_form(w, x, y)
    else:
        theta = gradient_descent(x, y, J, limit=150, alpha=0.1)

    for i in range(n):
        _learned[i] = theta[0] + theta[1] * x[i]

    return _learned


_x, _y = get_vector_data(ds2)

learned_cf = LWR(_x, _y, CLOSED_FORM)
learned_gd = LWR(_x, _y, GRADIENT_DESCENT)
# plot using linear regression

t0, t1 = pa.closed_form(*get_vector_data(ds2))

line_points = [0.5, 1]
line_values = [t0 + lp * t1 for lp in line_points]

plt.plot(_x, _y, 'ko', markersize=3, label="data set 2")
plt.plot(_x, learned_cf, markersize=1, label="Learned Closed Form")
plt.plot(_x, learned_gd, 'r', markersize=1, label="Learned Gradient Descent")
plt.plot(line_points, line_values, 'g', markersize=3, label="Learned Normal Regression")


plt.legend(loc='best')

plt.xlabel("height")
plt.ylabel("weight")

plt.show()

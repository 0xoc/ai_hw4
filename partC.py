import matplotlib.pyplot as plt
import numpy as np

# the two different classes in data
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


def load(file):
    handle = open(file)

    data = handle.read()

    data = data.split('\n')

    _x = []
    _y = []
    _class = []

    for line in data:
        f1, f2, _cls = tuple(line.split(','))
        _x.append(f1)
        _y.append(f2)
        _class.append(value[_cls])

    return _x, _y, _class


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost()

# load the training set
x, y, cls = load("training_set.data")

# class A indexes
class_A = [i for i in range(len(cls)) if cls[i] == value[A]]
# class B indexes
class_B = [i for i in range(len(cls)) if cls[i] == value[B]]

# plot class A data with red color
plt.plot([x[i] for i in class_A], [y[i] for i in class_A], color[A] + 'o', markersize=3)
# plot class B data with red color
plt.plot([x[i] for i in class_B], [y[i] for i in class_B], color[B] + 'o', markersize=3)

plt.legend(loc='best')
plt.show()

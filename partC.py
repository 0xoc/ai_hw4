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


# load the training set
x, y, cls = load("training_set.data")

class_A = [i for i in range(len(cls)) if cls[i] == value[A]]
class_B = [i for i in range(len(cls)) if cls[i] == value[B]]

print(class_A)

# plot class A data with red color
plt.plot([x[i] for i in class_A], [y[i] for i in class_A], color[A] + 'o')

# plot class B data with red color
plt.plot([x[i] for i in class_B], [y[i] for i in class_B], color[B] + 'o')

plt.show()

def load(file):
    handle = open(file)

    data = handle.read()

    data = data.split('\n')

    x = []
    y = []
    _class = []

    for line in data:
        f1, f2, cls = tuple(line.split(','))
        x.append(f1)
        y.append(f2)
        _class.append(cls)

    return x, y, _class


# load the training set
x, y, cls = load("training_set.data")

print(x, y, cls)

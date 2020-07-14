import matplotlib.pyplot as plt
import numpy as np

# the two different classes in data
from utils import load, value, A, B, color, make_data


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict(_features, _weights):
    z = np.dot(_features, _weights)
    d = sigmoid(z)
    return d


def cost_function(_features, _labels, _weights):
    observations = len(_labels)
    predictions = predict(_features, _weights)
    class1_cost = _labels * np.log(predictions)
    class2_cost = (1 - _labels) * np.log(1 - predictions)
    cost = class1_cost + class2_cost
    cost = cost.sum() / observations

    return cost


def update_weights(_features, _labels, _weights, lr):
    N = len(_features)
    predictions = predict(_features, _weights)

    gradient = np.dot(_features.T, predictions - _labels) / N
    _weights -= gradient * lr

    return _weights


def decision_boundary(prob):
    return 1 if prob >= .5 else 0


def classify(predictions):
    """
    input  - N element array of predictions between 0 and 1
    output - N element array of 0s (False) and 1s (True)
    """
    return np.array([decision_boundary(prb) for prb in predictions])


def train(_features, _labels, _weights, lr, iters):
    cost_history = []

    for i in range(iters):
        _weights = update_weights(_features, _labels, _weights, lr)

        # Calculate error for auditing purposes
        cost = cost_function(_features, _labels, _weights)
        cost_history.append(cost)

        # Log Progress
        if i % 10000 == 0:
            print("iter: " + str(i) + " cost: " + str(cost))

    return _weights, cost_history


def calculate_accuracy(data_set_file, _weights):
    # load the testing set, for later use
    x_test, y_test, labels_test = load(data_set_file)

    # calculate the accuracy
    N = len(labels_test)
    hit = 0

    for i in range(N):
        if decision_boundary(predict(np.array([1, x_test[i], y_test[i]]), weights)) == bool(labels_test[i]):
            hit += 1

    return hit / N * 100


# make training and test data randomly chosen from original dataset
make_data()

# load the training set
x, y, labels = load("training_set.data")

# class A indexes
class_A = [i for i in range(len(labels)) if labels[i] == value[A]]
# class B indexes
class_B = [i for i in range(len(labels)) if labels[i] == value[B]]

features = np.array([[1, x[i], y[i]] for i in range(len(labels))])
labels = np.array(labels)

# train on data
initial_weights = np.zeros(3)
weights, history = train(features, labels, initial_weights, 0.1, 100000)

# accuracy on test data
print("Model Accuracy on Test Data Set: %d percent" % calculate_accuracy('test_set.data', weights))
print("Model Accuracy on Training Data Set: %d percent" % calculate_accuracy('training_set.data', weights))

# getting the x co-ordinates of the decision boundary
plot_x = np.array([min(x) - 2, max(x) + 2])
# getting corresponding y co-ordinates of the decision boundary
plot_y = (-1 / weights[2]) * (weights[1] * plot_x + weights[0])

# plot class A data with red color
plt.plot([x[i] for i in class_A], [y[i] for i in class_A], color[A] + 'o', markersize=3, label=A)
# plot class B data with red color
plt.plot([x[i] for i in class_B], [y[i] for i in class_B], color[B] + 'o', markersize=3, label=B)

# plot class B data with red color
plt.plot(plot_x, plot_y, 'g', markersize=3, label="Boundary")

plt.legend(loc='best')
plt.show()

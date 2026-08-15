import random
import numpy as np


class Perceptron:

    def __init__(self, input_size):
        self.node_weights = []
        self.node_bias = []
        self.i = input_size

        # Matches length of inputs
        for n_w in range(0, input_size):
            self.node_weights.append((float(random.randrange(-100, 100)) / 100))

        self.node_bias.append((float(random.randrange(-100, 100)) / 100))

    def get_nodes(self):
        print(self.node_weights)

    def get_bias(self):
        print(self.node_bias)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, input_data):
        # Multiply each input by its weight, then add the bias
        layer = 0
        for x in range(0, len(input_data)):
            layer = layer + (input_data[x] * self.node_weights[x])

        layer = layer + self.node_bias[0]
        return self.sigmoid(layer)

    def train(self, xs, ys, learning_rate, epochs):
        for e in range(0, epochs):
            weight_gradients = []
            for w in range(0, len(self.node_weights)):
                weight_gradients.append(0.0)

            bias_gradient = 0.0

            for i, x in enumerate(xs):
                pred = self.predict(x)

                # Derivative of Sigmoid: pred * (1 - pred)
                sigmoid_derivative = pred * (1 - pred)
                error = (pred - ys[i]) * sigmoid_derivative

                # Accumulate gradients for weights without altering original list
                for j in range(0, len(self.node_weights)):
                    weight_gradients[j] = weight_gradients[j] + (error * x[j])

                bias_gradient = bias_gradient + error

            total_samples = len(xs)

            # Update weights using average gradient
            for j in range(0, len(self.node_weights)):
                average_weight_gradient = weight_gradients[j] / total_samples
                self.node_weights[j] = self.node_weights[j] - (learning_rate * average_weight_gradient)

            # Update bias using average gradient
            average_bias_gradient = bias_gradient / total_samples
            self.node_bias[0] = self.node_bias[0] - (learning_rate * average_bias_gradient)


# Corrected AND gate labels: [0,0]->0, [1,1]->1, [0,1]->0, [1,0]->0
x = [[0, 0], [1, 1], [0, 1], [1, 0]]
y = [0, 1, 0, 0]

p = Perceptron(2)

print("Initial Test")
for i in range(0, len(x)):
    print(f" Predicting {x[i]} = {p.predict(x[i])}")

print("Training Start")
# Increased learning rate and epochs for clear Sigmoid convergence
p.train(x, y, 0.5, 500)

print("After Training Test")
for i in range(0, len(x)):
    print(f" Predicting {x[i]} = {p.predict(x[i])}")
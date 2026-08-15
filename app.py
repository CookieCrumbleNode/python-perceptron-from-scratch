import random
import numpy as np


class Perceptron:

    def __init__(self, input_size):
        self.node_weights = []
        self.node_bias = []
        self.i = input_size

        # We need Weights for each expected Input
        for n_w in range(0, input_size):
            self.node_weights.append((float(random.randrange(-100, 100)) / 100))

        # We need a bias
        self.node_bias.append((float(random.randrange(-100, 100)) / 100))

    def get_nodes(self):
        # This was for troubleshooting
        print(self.node_weights)

    def get_bias(self):
        # This was for troubleshooting
        print(self.node_bias)

    def sigmoid(self, x):
        # Sigmoid activation function.
        return 1 / (1 + np.exp(-x))

    def predict(self, input_data):
        # This is our prediction function

        # Multiply each input by its weight, then add the bias
        layer = 0
        for x in range(0, len(input_data)):
            layer = layer + (input_data[x] * self.node_weights[x])

        # Adding our bias
        layer = layer + self.node_bias[0]

        # Returning the simoid for this single perceptron
        return self.sigmoid(layer)

    def train(self, xs, ys, learning_rate, epochs):
        # Training function - We need to set the input (x), output (y), learning rate and how many times we want to loop through the training

        for e in range(0, epochs):
            # Looping through the requested epochs

            # Storing the weight gradients
            weight_gradients = []

            for w in range(0, len(self.node_weights)):
                # Building our weight gradient array
                weight_gradients.append(0.0)

            bias_gradient = 0.0

            for i, x in enumerate(xs):
                # looping through the input training data and predicting the first element in the dataset
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


# Gate labels: [0,0] -> 0, [1,1] -> 1, [0,1] -> 0, [1,0] -> 0
x = [[0, 0], [1, 1], [0, 1], [1, 0]]
y = [0, 1, 0, 0]

# Creating a Perceptron which allows a designated number of inputs
p = Perceptron(2)

print("Test 1 - Pre-Training")
for i in range(0, len(x)):
    print(f" Predicting {x[i]} = {p.predict(x[i])}")

print("-----------");

Learning_Rate = 0.5
Epoch = 500

print(f"Training Start - Learning Rate: {Learning_Rate}, Epoch: {Epoch} ")
# Higher learning_rate and epoch to show clear results - This causes overfitting
p.train(x, y, Learning_Rate, Epoch)

print("-----------");

print("Test 2 - Post Training")
for i in range(0, len(x)):
    print(f" Predicting {x[i]} = {p.predict(x[i])}")
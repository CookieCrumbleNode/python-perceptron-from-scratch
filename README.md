# Python Perceptron — From Scratch

A simple implementation of a **single-neuron perceptron using Python and NumPy**, built from scratch to understand the fundamentals behind neural networks and machine learning.

This project was created as part of my journey into **Python programming, artificial intelligence, and machine learning**, with the goal of understanding what is happening inside a neural network rather than relying entirely on high-level ML libraries.

## Project Overview

This project implements a basic artificial neuron that:

* Accepts multiple input values
* Assigns a weight to each input
* Adds a bias
* Applies a sigmoid activation function
* Produces a prediction
* Calculates gradients using the sigmoid derivative
* Updates its weights and bias using gradient descent
* Learns to recognise the **AND logical gate**

The implementation does not use machine-learning frameworks such as TensorFlow or PyTorch. The training process is implemented manually to help demonstrate an understanding of the underlying concepts.

## How It Works

The perceptron calculates a weighted sum of its inputs:

```text
z = (x₁ × w₁) + (x₂ × w₂) + bias
```

The result is then passed through the sigmoid activation function:

```text
sigmoid(x) = 1 / (1 + e⁻ˣ)
```

This produces a value between `0` and `1`, which can be interpreted as the neuron's prediction.

During training, the model:

1. Makes a prediction for each training example.
2. Calculates the prediction error.
3. Calculates the gradient of the sigmoid function.
4. Accumulates gradients for the weights and bias.
5. Averages the gradients across the training data.
6. Updates the weights and bias using the learning rate.
7. Repeats the process for the specified number of epochs.

## Example Dataset

The model is trained on the AND logical gate:

| Input 1 | Input 2 | Expected Output |
| ------: | ------: | --------------: |
|       0 |       0 |               0 |
|       1 |       1 |               1 |
|       0 |       1 |               0 |
|       1 |       0 |               0 |

The goal is for the neuron to learn that the output should be `1` only when **both inputs are `1`**.

## Example

Before training, the randomly initialised weights and bias result in relatively poor predictions.

After training for 500 epochs, the predictions should move towards the expected values:

```text
Initial Test
Predicting [0, 0] = ...
Predicting [1, 1] = ...
Predicting [0, 1] = ...
Predicting [1, 0] = ...

Training Start

After Training Test
Predicting [0, 0] = ...
Predicting [1, 1] = ...
Predicting [0, 1] = ...
Predicting [1, 0] = ...
```

Because the model uses random initial weights, the exact starting and final prediction values can vary between runs.

## Technologies Used

* **Python**
* **NumPy**
* Object-oriented programming
* Basic calculus / derivatives
* Gradient descent
* Sigmoid activation
* Neural network fundamentals

## Project Structure

```text
perceptron/
│
├── perceptron.py
└── README.md
```

## Key Concepts Demonstrated

### Neural Network Fundamentals

The `Perceptron` class represents a very small neural network consisting of a single computational node.

The model contains:

```python
self.node_weights
self.node_bias
```

These parameters are modified during training so that the model can improve its predictions.

### Sigmoid Activation

The sigmoid function is implemented using NumPy:

```python
def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))
```

The sigmoid function converts the neuron's weighted sum into a value between `0` and `1`.

### Gradient Descent

The `train()` method manually calculates gradients and adjusts the model's parameters:

```python
self.node_weights[j] = (
    self.node_weights[j]
    - (learning_rate * average_weight_gradient)
)
```

This demonstrates the basic principle behind optimisation in neural networks: adjust parameters in the direction that reduces the model's error.

### Object-Oriented Python

The project uses a Python class to encapsulate the model's:

* Parameters
* Prediction logic
* Activation function
* Training algorithm

This helped me practise structuring Python code using classes and methods rather than writing everything procedurally.

## Why I Built This

I built this project to strengthen my understanding of **how neural networks work internally**.

Rather than starting with a high-level framework, I wanted to understand the individual steps involved in training a model:

```text
Input
  ↓
Weights
  ↓
Weighted Sum
  ↓
Bias
  ↓
Sigmoid
  ↓
Prediction
  ↓
Error
  ↓
Gradient
  ↓
Weight/Bias Update
  ↓
Repeat
```

Building the algorithm myself has helped me develop a stronger foundation before moving on to larger machine-learning libraries and projects.

## What I Learned

Through this project I gained practical experience with:

* Python classes and methods
* Lists and loops
* NumPy
* Mathematical functions
* Activation functions
* Derivatives
* Gradient descent
* Model parameters
* Training epochs
* Learning rates
* Debugging machine-learning code
* Understanding the relationship between predictions, errors and gradients

## Future Improvements

There are several areas I would like to improve as I continue developing the project.

* [ ] Add a loss function such as binary cross-entropy
* [ ] Add training/testing data separation
* [ ] Add configurable activation functions
* [ ] Add configurable loss functions
* [ ] Track and display loss during training
* [ ] Add matplotlib graphs showing training progress
* [ ] Add support for more than one neuron
* [ ] Implement a multi-layer neural network from scratch
* [ ] Add unit tests using `pytest`
* [ ] Improve input validation and error handling
* [ ] Experiment with different datasets
* [ ] Compare the implementation against a model built using scikit-learn

## Running the Project

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2. Install the dependency

```bash
pip install numpy
```

### 3. Run the program

```bash
python app.py
```

## What This Project Represents

This is a learning project rather than a production-ready machine-learning library.

The purpose is to demonstrate my progression in **Python development and AI/ML fundamentals**, including the ability to understand an algorithm, implement it from scratch, test it, and identify areas for future improvement.

As I continue learning, I plan to build increasingly complex projects involving **Python, machine learning, APIs, data processing and AI applications**.

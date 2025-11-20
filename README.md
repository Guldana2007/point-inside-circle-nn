# This is H1
**Point Inside Circle Neural Network**
**1. Introduction**

This project demonstrates the development and training of a simple feed-forward neural network (Multi-Layer Perceptron) designed to determine whether a point with coordinates (x, y) lies inside a circle of radius 5 centered at (0, 0).
A point belongs to the circle if the following condition is satisfied:

**x² + y² ≤ 25**

The model is trained on synthetically generated data and evaluated on a separate test set.
A helper function is also provided to classify individual points manually.

**2. Dataset Description**

A dataset of random points is generated within the square interval:

x ∈ [-6, 6]
y ∈ [-6, 6]

Each point is assigned a binary label based on its position:

**1 — the point lies inside the circle**

**0 — the point lies outside**

The dataset is then split into training and testing subsets using a standard 80/20 ratio.

**3. Neural Network Architecture**

The model architecture consists of the following layers:

Input layer: 2 numerical features (x, y)

Hidden layer 1: 16 neurons, ReLU activation

Hidden layer 2: 16 neurons, ReLU activation

Output layer: 1 neuron, Sigmoid activation

ReLU activation is used in the hidden layers due to its effectiveness for nonlinear problems and its resistance to the vanishing gradient issue.
Sigmoid activation is selected for the output layer because the task is a binary classification problem.

**4. Training Methodology**

Training is performed using the Backpropagation algorithm.
Key steps include:

**4.1 Forward Pass**

Each layer computes a weighted sum of inputs followed by the application of an activation function.

**4.2 Error Calculation**

The model's prediction is compared to the expected label to compute the error.

**4.3 Delta Computation**

The delta value for each neuron is determined as the product of the error and the derivative of the activation function.

**4.4 Weight Update**

Weights are updated according to the rule:

w_new = w_old + η × δ × input

where η is the learning rate.
TensorFlow handles all internal gradient computations automatically through the model.fit() procedure.

**5. Project Files**

circle_classifier.py — main Python implementation

circle_classifier.ipynb — Jupyter Notebook with explanations and visualizations

(Optional) requirements.txt — dependency list

**6. Execution Instructions**

To install dependencies:

pip install -r requirements.txt


To run the Python script:

python circle_classifier.py


To open the Jupyter Notebook:

jupyter notebook

**7. Example Results**

The trained model demonstrates high classification accuracy.
Typical outputs include:

Point (1.00, 1.00) -> inside the circle
Point (6.00, 0.00) -> outside the circle

**8. Purpose of the Project**

This project is intended for educational use and demonstrates fundamental concepts of neural networks, activation functions, and the Backpropagation learning algorithm.

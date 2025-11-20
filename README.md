Point Inside Circle Neural Network

This project demonstrates how a simple neural network (Multi-Layer Perceptron) can classify whether a point (x, y) lies inside a circle of radius 5 centered at (0, 0).

The mathematical condition for a point to be inside the circle is:

𝑥
2
+
𝑦
2
≤
25
x
2
+y
2
≤25
Project Description

The workflow includes the following steps:

Random points are generated within the range [-6, 6] × [-6, 6].

Each point is labeled as:

1 — if the point is inside the circle

0 — if it is outside

A neural network is built and trained to learn this classification boundary.

The model is evaluated on unseen test data.

A helper function is provided for manual point classification.

Model Architecture

A simple feed-forward neural network is used:

Input layer: 2 features (x, y)

Hidden layer: 16 neurons, ReLU activation

Hidden layer: 16 neurons, ReLU activation

Output layer: 1 neuron, Sigmoid activation

ReLU is used in hidden layers because it works well for nonlinear boundaries and avoids vanishing gradients.
Sigmoid is used in the output layer to produce a probability value between 0 and 1.

Training Algorithm (Backpropagation)

Training is performed using the Backpropagation algorithm.

Forward pass:
Each layer computes a weighted sum followed by an activation function.

Error calculation:

𝑒
𝑟
𝑟
𝑜
𝑟
=
𝑒
𝑥
𝑝
𝑒
𝑐
𝑡
𝑒
𝑑
−
𝑝
𝑟
𝑒
𝑑
𝑖
𝑐
𝑡
𝑒
𝑑
error=expected−predicted

Delta computation:

𝛿
=
𝑒
𝑟
𝑟
𝑜
𝑟
⋅
𝑓
′
(
𝑜
𝑢
𝑡
)
δ=error⋅f
′
(out)

where 
𝑓
′
f
′
 is the derivative of the activation function.

Weight update rule:

𝑤
𝑛
𝑒
𝑤
=
𝑤
𝑜
𝑙
𝑑
+
𝜂
⋅
𝛿
⋅
𝑖
𝑛
𝑝
𝑢
𝑡
w
new
	​

=w
old
	​

+η⋅δ⋅input

TensorFlow automatically handles all computations during training using model.fit().

Project Files

circle_classifier.py — main Python script

circle_classifier.ipynb — explanatory Jupyter Notebook with theory and visualizations

(Optional) You may include a requirements.txt file to install dependencies easily.

How to Run

Install dependencies:

pip install -r requirements.txt


Run the script:

python circle_classifier.py


Or open the notebook:

jupyter notebook

Example Output

A correctly trained model should classify points like:

Point (1.00, 1.00) -> inside
Point (6.00, 0.00) -> outside

Purpose

This project was created for educational purposes to illustrate basic concepts of neural networks, activation functions, and the Backpropagation training algorithm.

"""
circle_classifier.py

This script builds and trains a small neural network that answers a simple question:

If we take a point with coordinates (x, y),
is this point inside a circle of radius 5 with center at (0, 0) or outside?

A point is inside the circle if:
    x^2 + y^2 <= 25

Steps:
1. Create many random points in the square [-6, 6] x [-6, 6].
2. For each point, compute the correct answer: inside (1.0) or outside (0.0).
3. Train a neural network to learn this rule.
4. Evaluate the model on a test set.
5. Try a few example points by hand.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------
# 1. Make results reproducible
# ---------------------------------------------------------------------
# These lines fix the “randomness” so that each run gives the same result.
np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------------------------------------------------
# 2. Create training data
# ---------------------------------------------------------------------

# How many points we create
N = 20000

# X will have shape (N, 2): [x, y] for each point
X = np.random.uniform(-6, 6, size=(N, 2))


def point_in_circle(x: float, y: float, r: float = 5.0) -> float:
    """
    Return 1.0 if (x, y) is inside or on a circle with radius r, else 0.0.

    Rule: inside if x^2 + y^2 <= r^2.
    """
    return 1.0 if x**2 + y**2 <= r**2 else 0.0


# Build y: label for each point (1.0 inside, 0.0 outside)
y = np.array([point_in_circle(x, y) for x, y in X], dtype=np.float32)

print("Example point:", X[0])
print("Label (1=inside, 0=outside):", y[0])

# ---------------------------------------------------------------------
# 3. Split data into training and test parts
# ---------------------------------------------------------------------

# We want to know not only how well the network learns the training examples,
# but also how well it works on new unseen points.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

# ---------------------------------------------------------------------
# 4. Build the neural network
# ---------------------------------------------------------------------
# We use a small network:
# - input: 2 numbers (x, y);
# - hidden layer 1: 16 neurons with ReLU activation;
# - hidden layer 2: 16 neurons with ReLU;
# - output layer: 1 neuron with Sigmoid activation (probability from 0 to 1).

model = models.Sequential([
    # Input layer: two numbers (x and y)
    layers.Input(shape=(2,)),

    # Hidden layer 1
    layers.Dense(16, activation='relu'),

    # Hidden layer 2
    layers.Dense(16, activation='relu'),

    # Output: one number between 0 and 1 (probability)
    layers.Dense(1, activation='sigmoid')
])

# Tell Keras how we want to train the model
model.compile(
    optimizer='adam',              # how we update weights
    loss='binary_crossentropy',    # loss for 0/1 classification
    metrics=['accuracy']           # we want to see classification accuracy
)

print("\nModel summary:")
model.summary()

# ---------------------------------------------------------------------
# 5. Train the model
# ---------------------------------------------------------------------
# Keras (TensorFlow) does all hard work:
# 1. Sends data through the network (forward pass).
# 2. Compares predictions with correct answers (loss).
# 3. Spreads the error backwards (backpropagation).
# 4. Changes weights a little bit to make the loss smaller.
# We repeat this process for many epochs.

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,  # 20% of training data used as validation
    epochs=20,
    batch_size=32,
    verbose=1
)

# ---------------------------------------------------------------------
# 6. Check the model on the test set
# ---------------------------------------------------------------------

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

# ---------------------------------------------------------------------
# 7. Try some points manually
# ---------------------------------------------------------------------


def predict_point(x: float, y: float, threshold: float = 0.5) -> int:
    """
    Predict whether (x, y) is inside the circle.

    - x, y: coordinates of the point;
    - threshold: if probability >= threshold -> class 1, else class 0.
    """
    # Build a batch of one point
    point = np.array([[x, y]], dtype=np.float32)
    # Ask the model for probability that the point is inside the circle
    prob = model.predict(point, verbose=0)[0, 0]
    # Convert probability into class 0 or 1
    predicted_class = 1 if prob >= threshold else 0
    print(f"Point ({x:.2f}, {y:.2f}) -> prob_inside = {prob:.3f}, class = {predicted_class}")
    return predicted_class


if __name__ == "__main__":
    # Try a few points
    predict_point(1, 1)    # clearly inside
    predict_point(5, 0)    # on the circle border
    predict_point(6, 0)    # clearly outside
    predict_point(4, 4)    # outside

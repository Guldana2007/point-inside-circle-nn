"""
circle_classifier.py

Task:
Build and train a neural network that, given coordinates of a point (x, y),
predicts whether the point lies inside a circle of radius 5 centered at (0, 0).

A point is inside the circle if:
    x^2 + y^2 <= 25

The script:
- generates a synthetic dataset (points in [-6, 6] x [-6, 6]),
- labels each point as inside / outside the circle,
- defines and trains a neural network (MLP),
- evaluates the model on a test set,
- provides a helper function to test individual points.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------
# 1. Reproducibility: fix random seeds
# ---------------------------------------------------------------------
np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------------------------------------------------
# 2. Data generation and labeling
# ---------------------------------------------------------------------

# Number of random points in the square [-6, 6] x [-6, 6]
N = 20000


def point_in_circle(x: float, y: float, r: float = 5.0) -> float:
    """
    Return 1.0 if the point (x, y) lies inside or on a circle of radius r,
    0.0 otherwise.

    Mathematically:
        inside if x^2 + y^2 <= r^2
    """
    return 1.0 if x**2 + y**2 <= r**2 else 0.0


# Generate N two-dimensional points
X = np.random.uniform(-6, 6, size=(N, 2))

# Create labels: 1.0 for inside, 0.0 for outside
y = np.array([point_in_circle(x, y) for x, y in X], dtype=np.float32)

print("Sample point:", X[0])
print("Its label:", y[0])

# ---------------------------------------------------------------------
# 3. Train–test split
# ---------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

# ---------------------------------------------------------------------
# 4. Neural network architecture
# ---------------------------------------------------------------------
# Model: simple MLP with two hidden layers and sigmoid output.

model = models.Sequential([
    # Input layer: expects vectors of size 2 (x and y)
    layers.Input(shape=(2,)),

    # Hidden layer 1: 16 neurons, ReLU activation
    layers.Dense(16, activation='relu'),

    # Hidden layer 2: 16 neurons, ReLU activation
    layers.Dense(16, activation='relu'),

    # Output layer: 1 neuron, Sigmoid activation
    # Outputs probability that the point is inside the circle
    layers.Dense(1, activation='sigmoid')
])

# Compile the model:
# - optimizer Adam
# - binary cross-entropy loss (for binary classification)
# - accuracy metric
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nModel summary:")
model.summary()

# ---------------------------------------------------------------------
# 5. Training (Backpropagation handled by Keras)
# ---------------------------------------------------------------------
# Keras internally performs:
# 1) forward pass,
# 2) loss computation,
# 3) backward pass (backpropagation),
# 4) weight updates with Adam optimizer.

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,  # 20% of training data used for validation
    epochs=20,
    batch_size=32,
    verbose=1
)

# ---------------------------------------------------------------------
# 6. Evaluation on the test set
# ---------------------------------------------------------------------

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

# ---------------------------------------------------------------------
# 7. Helper function to test individual points
# ---------------------------------------------------------------------


def predict_point(x: float, y: float, threshold: float = 0.5) -> int:
    """
    Predict whether a single point (x, y) is inside the circle.

    Steps:
    1) Build a batch of shape (1, 2).
    2) Use the trained model to compute probability that the point is inside.
    3) Compare the probability with the threshold and return:
       - 1 if prob >= threshold (inside),
       - 0 otherwise (outside).
    """
    point = np.array([[x, y]], dtype=np.float32)
    prob = model.predict(point, verbose=0)[0, 0]
    predicted_class = 1 if prob >= threshold else 0

    print(
        f"Point ({x:.2f}, {y:.2f}) -> "
        f"prob_inside = {prob:.3f}, predicted_class = {predicted_class}"
    )
    return predicted_class


# ---------------------------------------------------------------------
# 8. Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Some example points to check the model's predictions
    predict_point(1.0, 1.0)   # clearly inside
    predict_point(5.0, 0.0)   # on the boundary
    predict_point(6.0, 0.0)   # outside
    predict_point(4.0, 4.0)   # outside

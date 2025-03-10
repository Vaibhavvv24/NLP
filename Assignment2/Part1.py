import numpy as np
import pandas as pd

# Activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# MLP model class
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output, lr):
        m = X.shape[0]

        # Output layer error
        d_output = (output - y) * sigmoid_derivative(output)

        # Hidden layer error
        d_hidden = np.dot(d_output, self.W2.T) * relu_derivative(self.a1)

        # Update weights and biases
        self.W2 -= lr * np.dot(self.a1.T, d_output) / m
        self.b2 -= lr * np.sum(d_output, axis=0, keepdims=True) / m

        self.W1 -= lr * np.dot(X.T, d_hidden) / m
        self.b1 -= lr * np.sum(d_hidden, axis=0, keepdims=True) / m

    def train(self, X, y, epochs, lr):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, lr)

    def predict(self, X):
        output = self.forward(X)
        return (output > 0.5).astype(int)

# Load dataset
data = pd.read_csv('/kaggle/input/exclusive-xor-dataset/xor.csv')

# Extract input features (X1, X2) and output labels
X = data[['X1', 'X2']].values  # Shape will be (n_samples, 2)
y = data['label'].values.reshape(-1, 1)  # Shape will be (n_samples, 1)

print("Input shape:", X.shape)  # Should be (n_samples, 2)
print("Label shape:", y.shape)  # Should be (n_samples, 1)
print(X.shape)
# # K-Fold Cross Validation
from sklearn.model_selection import KFold

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

accuracies = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"Training Fold {fold + 1}...")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Create and train the MLP model
    mlp = MLP(input_size=2, hidden_size=128, output_size=1)
    mlp.train(X_train, y_train, epochs=50000, lr=0.05)

    # Evaluate the model
    y_pred = mlp.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    accuracies.append(accuracy)

    print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")

print(f"Average Accuracy: {np.mean(accuracies):.4f}")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer_data = load_breast_cancer()
X = cancer_data.data
y = cancer_data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
std[std == 0] = 1

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic(X, y, lam=0, penalty=None, learning_rate=0.1, epochs=1000):
    m = X.shape[0]
    n = X.shape[1]

    theta1 = np.zeros(n)
    theta0 = 0

    for epoch in range(epochs):
        z = np.dot(X, theta1) + theta0
        y_hat = sigmoid(z)

        error = y_hat - y

        d_theta1 = np.dot(X.T, error) / m
        d_theta0 = np.sum(error) / m

        if penalty == "l2":
            d_theta1 += (lam / m) * theta1

        elif penalty == "l1":
            d_theta1 += (lam / m) * np.sign(theta1)

        theta1 -= learning_rate * d_theta1
        theta0 -= learning_rate * d_theta0

    return theta1, theta0

def accuracy(X, y, theta1, theta0):
    y_hat = sigmoid(np.dot(X, theta1) + theta0)
    predictions = (y_hat >= 0.5).astype(int)
    return np.mean(predictions == y)

theta1_mle, theta0_mle = train_logistic(X_train, y_train)

theta1_l2, theta0_l2 = train_logistic(
    X_train, y_train, lam=1, penalty="l2"
)

theta1_l1, theta0_l1 = train_logistic( X_train, y_train, lam=1, penalty="l1" )

print("MLE Accuracy:", accuracy(X_test, y_test, theta1_mle, theta0_mle))
print("MAP L2 Accuracy:", accuracy(X_test, y_test, theta1_l2, theta0_l2))
print("MAP L1 Accuracy:", accuracy(X_test, y_test, theta1_l1, theta0_l1))

print("\nMLE weight norm:", np.linalg.norm(theta1_mle))
print("MAP L2 weight norm:", np.linalg.norm(theta1_l2))
print("MAP L1 weight norm:", np.linalg.norm(theta1_l1))

weights_mle = theta1_mle
weights_map_l2 = theta1_l2
weights_map_l1 = theta1_l1

features = np.arange(len(weights_mle))
width = 0.25

plt.figure(figsize=(14, 7))

plt.bar(features - width, weights_mle, width, label="MLE")
plt.bar(features, weights_map_l2, width, label="L2 MAP")
plt.bar(features + width, weights_map_l1, width, label="L1 MAP")

plt.xlabel("Feature Index")
plt.ylabel("Parameter Value")
plt.title("Comparison of Parameter Estimates")
plt.xticks(features)
plt.legend()
plt.grid(axis="y")

plt.show()

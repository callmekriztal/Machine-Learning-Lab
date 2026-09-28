import time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X, y = fetch_openml(
    "Fashion-MNIST",
    version=1,
    return_X_y=True,
    as_frame=False
)

y = y.astype(np.int8)
X = X / 255.0

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

X_train, y_train = X_train[:10000], y_train[:10000]
X_test, y_test = X_test[:2000], y_test[:2000]

models = {
    "Linear": SVC(kernel="linear"),
    "Polynomial": SVC(kernel="poly", degree=3),
    "RBF": SVC(kernel="rbf")
}

results = {}

for name in models:

    start = time.time()

    model = models[name]
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    train_time = time.time() - start
    accuracy = accuracy_score(y_test, predictions)

    results[name] = [accuracy, train_time]

print("\nSVM Performance")

for name in results:
    accuracy = results[name][0]
    train_time = results[name][1]

    print(
        f"{name:12} "
        f"Accuracy: {accuracy:.4f}   "
        f"Training Time: {train_time:.2f}s"
    )

best_name = ""
best_accuracy = 0

for name in results:

    accuracy = results[name][0]

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_name = name

print("\nBest Model")
print(f"Kernel   : {best_name}")
print(f"Accuracy : {best_accuracy:.4f}")

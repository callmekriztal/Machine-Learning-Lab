import time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

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

kernels = {
    "Linear": SVC(kernel="linear"),
    "Polynomial": SVC(kernel="poly", degree=3),
    "RBF": SVC(kernel="rbf")
}

results = {}

for name, model in kernels.items():

    start = time.time()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    train_time = time.time() - start
    accuracy = accuracy_score(y_test, predictions)

    results[name] = (accuracy, train_time)

    print(f"\n{name} Kernel")
    print(f"Accuracy     : {accuracy:.4f}")
    print(f"Training Time: {train_time:.2f} seconds")
    print(classification_report(y_test, predictions))

print("\nPerformance Comparison")
print("-" * 45)

for kernel, (accuracy, time_taken) in results.items():
    print(
	f"{kernel:12} "
        f"Accuracy: {accuracy:.4f}  "
        f"Time: {time_taken:.2f}s"
    )

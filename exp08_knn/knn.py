import time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False)
y = y.astype(int)

X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, y_train = X_train[:5000], y_train[:5000]
X_test, y_test = X_test[:1000], y_test[:1000]

k_values = [1, 3, 5, 7, 9]

results = {}

for k in k_values:
    start = time.time()

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    end = time.time()

    results[k] = acc

    print(f"K={k}  Accuracy={acc:.4f}  Time={end-start:.2f}s")

best_k = max(results, key=results.get)
best_accuracy = results[best_k]

print("\nBest K:", best_k)
print(f"Best Accuracy: {best_accuracy:.4f}")

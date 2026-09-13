import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data[:, [2, 3]]
y = (iris.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel="linear", C=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

w = model.coef_[0]
b = model.intercept_[0]

x = np.linspace(
    X_train[:, 0].min() - 1,
    X_train[:, 0].max() + 1,
    100
)

boundary = -(w[0] * x + b) / w[1]
margin1 = -(w[0] * x + b - 1) / w[1]
margin2 = -(w[0] * x + b + 1) / w[1]

plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
            label="Setosa")
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
            label="Non-Setosa")

plt.plot(x, boundary, label="Decision Boundary")
plt.plot(x, margin1, "--", label="Margin")
plt.plot(x, margin2, "--")

plt.scatter(model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            facecolors="none", edgecolors="black",
            label="Support Vectors")

plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Linear SVM - Iris Dataset")
plt.legend()
plt.show()

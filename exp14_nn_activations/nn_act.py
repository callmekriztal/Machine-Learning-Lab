import time

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


mnist = fetch_openml(
    name="mnist_784",
    version=1,
    as_frame=False
)

X = mnist.data
y = mnist.target.astype(int)

print("Dataset shape:", X.shape)

X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

activations = ["logistic", "relu", "tanh"]

results = {}

for activation in activations:

    print("\nTraining with", activation, "activation...")

    model = MLPClassifier(
        hidden_layer_sizes=(128,),
        activation=activation,
        solver="adam",
        max_iter=20,
        random_state=42
    )

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    results[activation] = {
        "accuracy": accuracy,
        "training_time": training_time,
        "iterations": model.n_iter_
    }

    print("Training time:", round(training_time, 2), "seconds")
    print("Iterations:", model.n_iter_)
    print("Accuracy:", round(accuracy, 4))


print("\n----- COMPARISON -----")

for activation, result in results.items():

    print(
        activation,
        "-> Accuracy:",
        round(result["accuracy"], 4),
        "| Training time:",
        round(result["training_time"], 2),
        "seconds",
        "| Iterations:",
        result["iterations"]
    )


best_activation = max(
    results,
    key=lambda x: results[x]["accuracy"]
)

print("\n----- BEST RESULT -----")
print("Best Activation:", best_activation)
print(
    "Best Accuracy:",
    round(results[best_activation]["accuracy"], 4)
)

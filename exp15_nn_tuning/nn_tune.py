import time
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


fashion = fetch_openml(
    name="Fashion-MNIST",
    version=1,
    as_frame=False
)

X = fashion.data.astype("float32") / 255.0
y = fashion.target.astype(int)

print("Dataset shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

learning_rates = [0.001, 0.01]
batch_sizes = [32, 128]
epochs = [10, 20]

results = []

classes = np.unique(y_train)

for lr in learning_rates:
    for batch_size in batch_sizes:
        for epoch_count in epochs:

            print(
                "\nLearning rate:", lr,
                "| Batch size:", batch_size,
                "| Epochs:", epoch_count
            )

            model = MLPClassifier(
                hidden_layer_sizes=(128,),
                activation="relu",
                solver="sgd",
                learning_rate="constant",
                learning_rate_init=lr,
                batch_size=batch_size,
                max_iter=1,
                random_state=42
            )

            start_time = time.time()

            for epoch in range(epoch_count):
                model.partial_fit(
                    X_train,
                    y_train,
                    classes=classes
                )

            training_time = time.time() - start_time

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)

            results.append(
                (lr, batch_size, epoch_count, accuracy, training_time)
            )

            print("Accuracy:", round(accuracy, 4))
            print("Training time:", round(training_time, 2), "seconds")


print("\n----- RESULTS -----")

best_accuracy = 0
best_lr = 0
best_batch_size = 0
best_epochs = 0
best_time = 0

for result in results:

    if result[3] > best_accuracy:
        best_accuracy = result[3]
        best_lr = result[0]
        best_batch_size = result[1]
        best_epochs = result[2]
        best_time = result[4]


print("\n----- BEST RESULT -----")
print("Learning Rate:", best_lr)
print("Batch Size:", best_batch_size)
print("Epochs:", best_epochs)
print("Best Accuracy:", round(best_accuracy, 4))
print("Training Time:", round(best_time, 2), "seconds")


from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

wine = fetch_openml(
    name="wine-quality-red",
    version=1,
    as_frame=False
)

X = wine.data
y = wine.target.astype(int)

print("Dataset shape:", X.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


architectures = {
    "1 Hidden Layer - 10 neurons": (10,),
    "1 Hidden Layer - 50 neurons": (50,),
    "2 Hidden Layers - 50, 25 neurons": (50, 25),
    "3 Hidden Layers - 100, 50, 25 neurons": (100, 50, 25)
}

best_accuracy = 0
best_architecture = ""

for name, architecture in architectures.items():

    model = MLPClassifier(
        hidden_layer_sizes=architecture,
        activation="relu",
        solver="adam",
        max_iter=2000,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(name, ":", round(accuracy, 4))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_architecture = name

print("\nBest Model")
print("Best Architecture:", best_architecture)
print("Best Accuracy:", round(best_accuracy, 4))

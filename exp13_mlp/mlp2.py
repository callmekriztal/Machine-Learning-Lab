from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

wine = fetch_openml(name='wine-quality-red', version=1, as_frame=False)

X = wine.data
y = wine.target.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

architectures = [(8,), (16, 8), (24, 16, 8)]
results = []

for layers in architectures:
    model = MLPRegressor(
        hidden_layer_sizes=layers,
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append((layers, mse))

    print("Architecture:", layers)
    print("MSE:", round(mse, 3))
    print("R2 Score:", round(r2, 3))
    print()

best = results[0]

for result in results:
    if result[1] < best[1]:
        best = result

print("Best Architecture:", best[0])
print("Lowest MSE:", round(best[1], 3))

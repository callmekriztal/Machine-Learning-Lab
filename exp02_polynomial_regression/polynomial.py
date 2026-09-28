import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = fetch_openml("autoMpg", version=1, as_frame=False)

# displacement = column 2, mpg = column 0
X = data.data[:, 2].astype(float).reshape(-1, 1)
y = data.data[:, 0].astype(float)

# Remove missing values
mask = ~np.isnan(X[:, 0]) & ~np.isnan(y)
X, y = X[mask], y[mask]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
linear = LinearRegression()
linear.fit(X_train, y_train)
pred = linear.predict(X_test)

print("Linear Regression")
print("MSE:", mean_squared_error(y_test, pred))
print("R2:", r2_score(y_test, pred))

# Polynomial Regression
for degree in [2, 3, 4]:
    poly = PolynomialFeatures(degree)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    pred = model.predict(X_test_poly)

    print(f"\nPolynomial Degree {degree}")
    print("MSE:", mean_squared_error(y_test, pred))
    print("R2:", r2_score(y_test, pred))

# Plot
x = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

plt.scatter(X, y, alpha=0.5, label="Data")

# Linear Regression line
plt.plot(
    x,
    linear.predict(x),
    label="Linear Regression"
)

# Polynomial Regression curves
for degree in [2, 3, 4]:
    poly = PolynomialFeatures(degree)

    model = LinearRegression()
    model.fit(poly.fit_transform(X), y)

    plt.plot(
        x,
        model.predict(poly.transform(x)),
        label=f"Degree {degree}"
    )

plt.xlabel("Displacement")
plt.ylabel("MPG")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()

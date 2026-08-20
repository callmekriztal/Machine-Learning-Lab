import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
housing = fetch_california_housing(as_frame=True)

# Single feature: Average number of rooms
X = housing.data[['AveRooms']]
y = housing.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Visualization
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, alpha=0.5, label="Actual Data")

x_line = np.linspace(X.min().values[0], X.max().values[0], 100).reshape(-1, 1)
y_line = model.predict(x_line)

plt.plot(x_line, y_line, color="red", linewidth=2, label="Regression Line")

plt.xlabel("Average Rooms (AveRooms)")
plt.ylabel("Median House Value")
plt.title("Linear Regression on California Housing Dataset")
plt.legend()
plt.grid(True)
plt.show()

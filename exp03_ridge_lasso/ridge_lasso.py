import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def evaluate(model, name):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n", name)
    print("MSE:", mse)
    print("R-squared:", r2)
    return mse

linear_model = LinearRegression()
linear_mse = evaluate(linear_model, "Linear Regression")

ridge = Ridge()

ridge_params = { "alpha": [0.01, 0.1, 1, 10, 100] }

ridge_cv = GridSearchCV(
    ridge,
    ridge_params,
    cv=5,
    scoring="neg_mean_squared_error"
)

ridge_cv.fit(X_train_scaled, y_train)

print("\nBest Ridge Alpha:", ridge_cv.best_params_["alpha"])

ridge_model = ridge_cv.best_estimator_
ridge_mse = evaluate(ridge_model, "Ridge Regression")

lasso = Lasso(max_iter=10000)

lasso_params = { "alpha": [0.001, 0.01, 0.1, 1, 10] }

lasso_cv = GridSearchCV(
    lasso,
    lasso_params,
    cv=5,
    scoring="neg_mean_squared_error"
)

lasso_cv.fit(X_train_scaled, y_train)

print("\nBest Lasso Alpha:", lasso_cv.best_params_["alpha"])

lasso_model = lasso_cv.best_estimator_
lasso_mse=evaluate(lasso_model, "Lasso Regression")

if linear_mse < ridge_mse and linear_mse < lasso_mse:
    print("Best Model: Linear Regression")
elif ridge_mse < linear_mse and ridge_mse < lasso_mse:
    print("Best Model: Ridge Regression")
else:
    print("Best Model: Lasso Regression")

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import numpy as np
import matplotlib.pyplot as plt

housing = fetch_california_housing()

X = housing.data[:,2]
y = housing.target

z = (X - np.mean(X))/np.std(X)

mask = np.abs(z) <= 3

X = X[mask]
y = y[mask]

X_train,X_test,y_train,y_test = train_test_split(
	X,y,test_size = 0.2,random_state = 42)

mean = np.mean(X_train)
std = np.std(X_train)

X_train = (X_train - mean)/std
X_test = (X_test - mean)/std

theta0 = 0
theta1 = 0

lr = 0.01
epochs = 1000

m = len(X_train)

for i in range(epochs):
	y_pred = theta0 + theta1*X_train
	d_theta0 = (1/m)*np.sum((y_pred-y_train))
	d_theta1 = (1/m)*np.sum((y_pred-y_train)*X_train)
	theta0 -= lr*d_theta0
	theta1 -= lr*d_theta1

y_pred = theta0 + theta1 * X_test

mse = np.mean((y_test - y_pred)**2)
ss_res = np.sum((y_test - y_pred)**2)
ss_total = np.sum((y_test - np.mean(y_test))**2)

r2 = 1 - ss_res/ss_total

print("Gradient Descent")
print("Theta0 =", theta0)
print("Theta1 =", theta1)
print("MSE =", mse)
print("R2 =", r2)


plt.scatter(X_test,y_test)

X_line = np.sort(X_test)

plt.plot(X_line,theta0+theta1*X_line,color = "green")
plt.show()


X_train_ne = np.column_stack((np.ones(len(X_train)),X_train))
X_test_ne = np.column_stack((np.ones(len(X_test)),X_test))

theta = np.linalg.inv(X_train_ne.T @ X_train_ne) @ X_train_ne.T @ y_train

y_pred = X_test_ne  @ theta

mse = np.mean((y_test - y_pred)**2)
ss_res = np.sum((y_test - y_pred)**2)
ss_tot = np.sum((y_test -np.mean(y_test))**2)

print("\nNormal Equation")
print("Theta0 =", theta[0])
print("Theta1 =", theta[1])
print("MSE =", mse)
print("R2 =", r2)

plt.scatter(X_test,y_test)
X_line = np.sort(X_test)
X_line_ne = np.column_stack((np.ones(len(X_line)),X_line))
plt.plot(X_line,X_line_ne @ theta,color = "red")
plt.show()

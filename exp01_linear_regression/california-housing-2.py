import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
# Feature: 'AveRooms' (column index 2)
# Target: Median House Value (in $100,000s)
print(housing.feature_names,housing.target_names)
X = housing.data[:, 2].reshape(-1,1) #Convert to vectors
y = housing.target.reshape(-1,1)

# Filter structural outliers (AveRooms > 10) for cleaner modeling & visualization

valid_mask = X.flatten() <= 10
X = X[valid_mask]
y = y[valid_mask]

m,n = X.shape
X = np.hstack((np.ones((m,1)),X))
Beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
beta0,beta1 = Beta.flatten()
print("The regression coefficeients by normal equation are:",beta0,beta1)
#GD
Theta = np.zeros((n+1,1))
iter,eta = 15000,0.01 #set no of iterations and learning rate 
for i in range(iter):
    y_pred = X.dot(Theta) #prediction
    grad_theta = (1/m)*X.T.dot(y_pred-y) #gradient
    Theta -= eta*grad_theta
theta0,theta1 = Theta.flatten()
print("The regression coefficeients by gradient descent are:",theta0,theta1) 
#Metrics
y_pred_ne,y_pred_gd = X.dot(Beta),X.dot(Theta)
mse_ne,mse_gd = np.mean((y-y_pred_ne)**2),np.mean((y-y_pred_gd)**2)
print(f"The MSE for Normal Equation is={mse_ne} and for Gradient Descent={mse_gd}")
#for r2
ssr_ne = np.sum((y-y_pred_ne)**2)
sst = np.sum((y-np.mean(y))**2)
r2_ne = 1 - (ssr_ne/sst)
ssr_gd = np.sum((y-y_pred_gd)**2)
r2_gd = 1 - (ssr_gd/sst)
print(f"The R^2 scores are:for Normal Equation={r2_ne}, for Gradient Descent={r2_gd}")
#plt.scatter(X[:,1], y, color='royalblue', alpha=0.15, s=6, label='Actual Data Distribution')
plt.plot(X[:,1],y_pred_gd,'r',label='GD')
plt.plot(X[:,1],y_pred_ne,'y:',label='NE')
plt.scatter(X[:,1],y,s=2)#,alpha=.2
plt.legend(loc='upper right')
plt.show()

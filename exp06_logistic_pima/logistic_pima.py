import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

diabetes = fetch_openml(name='diabetes', version=1, as_frame=False)

X = diabetes.data
y = diabetes.target

print(diabetes.DESCR)

y = np.where(y == 'tested_positive', 1, 0)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true,y_pred)
    print("Accuracy :", acc)
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1-score :", f1_score(y_true, y_pred))
    return acc

model1 = LogisticRegression(max_iter=1000)
model1.fit(X_train, y_train)

y_pred1 = model1.predict(X_test)

print("Without feature scaling")

acc_wfs = evaluate(y_test, y_pred1)

mean = np.mean(X_train,axis=0)
std = np.std(X_train,axis=0)

std[std == 0] = 1

X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std

model2 = LogisticRegression(max_iter=1000)
model2.fit(X_train_scaled, y_train)

y_pred2 = model2.predict(X_test_scaled)

print("\nWith feature scaling")
acc_fs = evaluate(y_test, y_pred2)

print("\nPerformance")
if(acc_wfs > acc_fs ):
    print("Without Feature Scaling is better")
else:
    print("With Feature Scaling is Better")

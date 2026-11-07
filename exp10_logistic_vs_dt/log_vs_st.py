import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = fetch_openml("adult", version=2, as_frame=False)

X = data.data
y = data.target

mask = np.all(X != "?", axis=1)

X = X[mask]
y = y[mask]

y = (y == ">50K").astype(int)

categorical = [1, 3, 5, 6, 7, 8, 9, 13]

encoder = ColumnTransformer(
    [("onehot", OneHotEncoder(handle_unknown="ignore"),
      categorical)],
    remainder="passthrough"
)

X = encoder.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)

logistic = LogisticRegression(max_iter=1000)
logistic.fit(X_train, y_train)
logistic_pred = logistic.predict(X_test)

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)

print("LOGISTIC REGRESSION")

print("Accuracy  :", accuracy_score(y_test, logistic_pred))
print("Precision :", precision_score(y_test, logistic_pred))
print("Recall    :", recall_score(y_test, logistic_pred))
print("F1 Score  :", f1_score(y_test, logistic_pred))


print("\nDECISION TREE")

print("Accuracy  :", accuracy_score(y_test, tree_pred))
print("Precision :", precision_score(y_test, tree_pred))
print("Recall    :", recall_score(y_test, tree_pred))
print("F1 Score  :", f1_score(y_test, tree_pred))

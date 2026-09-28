import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = fetch_openml("adult", version=2, as_frame=False)

X = data.data
y = data.target
feature_names = list(data.feature_names)

X_str = X.astype(str)
missing = (X_str == "?").any(axis=1) | (X_str == "nan").any(axis=1)
X = X[~missing]
y = y[~missing]

y = np.char.replace(y.astype(str), ".", "")
y = np.char.strip(y)
y = np.where(y == "<=50K", 0, 1).astype(int)

numeric_names = ["age", "education-num", "capital-gain",
                 "capital-loss", "hours-per-week"]

numeric_idx = [feature_names.index(n) for n in numeric_names]
categorical_idx = [i for i in range(len(feature_names))
                   if i not in numeric_idx]

X_numeric = X[:, numeric_idx].astype(float)
X_categorical = X[:, categorical_idx].astype(str)

X_final = np.hstack([X_numeric, X_categorical])

num_cols = list(range(len(numeric_idx)))
cat_cols = list(range(len(numeric_idx),
                       len(numeric_idx) + len(categorical_idx)))

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)
logistic_pred = logistic_model.predict(X_test)

tree_model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=6,
    random_state=42
)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)

print("\nLogistic Regression")
print("Accuracy :", accuracy_score(y_test, logistic_pred))
print("Precision:", precision_score(y_test, logistic_pred))
print("Recall :", recall_score(y_test, logistic_pred))
print("F1 Score :", f1_score(y_test, logistic_pred))

print("\nDecision Tree")
print("Accuracy :", accuracy_score(y_test, tree_pred))
print("Precision:", precision_score(y_test, tree_pred))
print("Recall :", recall_score(y_test, tree_pred))
print("F1 Score :", f1_score(y_test, tree_pred))


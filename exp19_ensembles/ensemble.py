import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].copy()
y = df['Survived']

X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
X['Age'] = X['Age'].fillna(X['Age'].median())
X['Fare'] = X['Fare'].fillna(X['Fare'].median())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

bag = BaggingClassifier(estimator=DecisionTreeClassifier(),n_estimators=50,random_state=42)
bag.fit(X_train, y_train)
bag_pred = bag.predict(X_test)

boost = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),n_estimators=50,random_state=42)
boost.fit(X_train, y_train)
boost_pred = boost.predict(X_test)

for name, pred in [('Bagging', bag_pred), ('Boosting', boost_pred)]:
    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred, zero_division=0))
    print("Recall:", recall_score(y_test, pred, zero_division=0))
    print("F1-score:", f1_score(y_test, pred, zero_division=0))

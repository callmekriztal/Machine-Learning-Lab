import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from ucimlrepo import fetch_ucirepo

online_retail = fetch_ucirepo(id=352)
df = online_retail.data.original.copy()

print("Original shape:", df.shape)

df = df.sample(n=10000, random_state=42)

df = df.dropna(subset=["CustomerID"])

df = df[
    ~df["InvoiceNo"].astype(str).str.startswith("C")
]

df = df[
    (df["Quantity"] > 0) &
    (df["UnitPrice"] > 0)
]

df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

print("After preprocessing:", df.shape)

customer_data = df.groupby("CustomerID").agg(
    TotalSpent=("TotalAmount", "sum"),
    TotalQuantity=("Quantity", "sum"),
    NumInvoices=("InvoiceNo", "nunique"),
    UniqueProducts=("StockCode", "nunique"),
    AvgOrderValue=("TotalAmount", "mean")
).reset_index()

customer_data["PurchaseFrequency"] = (
    customer_data["NumInvoices"] /
    customer_data["NumInvoices"].max()
)

customer_data["Segment"] = pd.qcut(
    customer_data["TotalSpent"],
    3,
    labels=["Low", "Medium", "High"],
    duplicates="drop"
)

features = [
    "TotalQuantity",
    "NumInvoices",
    "UniqueProducts",
    "AvgOrderValue",
    "PurchaseFrequency"
]

X = customer_data[features]
y = customer_data["Segment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(14, 7))

plot_tree(
    model,
    feature_names=features,
    class_names=["Low", "Medium", "High"],
    filled=False,       # NO COLORS
    rounded=False,      # SIMPLE BOXES
    impurity=False,     # REMOVE ENTROPY/GINI
    proportion=False,
    precision=2,
    fontsize=8
)

plt.title("ID3 Decision Tree")
plt.tight_layout()
plt.show()

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(
    "Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree



df = pd.read_csv("OnlineRetail.csv", encoding="ISO-8859-1")


print("Original shape:", df.shape)
#print(df.head())


df = df.dropna(subset=["CustomerID"])
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]


df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]


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

print("\nCustomer data:")
#print(customer_data.head())


customer_data["Segment"] = pd.qcut(
    customer_data["TotalSpent"],
    q=3,
    labels=["Low", "Medium", "High"]
)

print("\nSegment distribution:")
print(customer_data["Segment"].value_counts())


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


id3_model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4,
    random_state=42
)

id3_model.fit(X_train, y_train)




y_pred = id3_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
#print(classification_report(y_test, y_pred))


plt.figure(figsize=(20, 10))

plot_tree(
    id3_model,
    feature_names=features,
    class_names=["Low", "Medium", "High"],
    #filled=True,
    rounded=True,
    fontsize=10,
    impurity=False
)

plt.title("ID3 Decision Tree for Customer Segmentation")
plt.show()


# --------------------------------------------------
# 10. FEATURE IMPORTANCE
# --------------------------------------------------

importance = pd.DataFrame({
    "Feature": features,
    "Importance": id3_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)


plt.figure(figsize=(8, 5))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance - ID3 Decision Tree")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

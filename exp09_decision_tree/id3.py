import numpy as np
import pandas as pd # type: ignore
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

data = fetch_ucirepo(id=352)
df = data.data.original.copy()

print("Original Dataset Shape:", df.shape)

df = df.dropna(subset=["CustomerID"]).copy()
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

print("Cleaned Dataset Shape:", df.shape)

reference_date = df["InvoiceDate"].max()

customer = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    TotalQuantity=("Quantity", "sum"),
    TotalSpend=("TotalPrice", "sum"),
    UniqueProducts=("StockCode", "nunique")
).reset_index()

print("\nCustomer Dataset:")
print(customer.head())
print("\nNumber of Customers:", len(customer))

customer["Segment"] = pd.qcut(
    customer["TotalSpend"],
    q=3,
    labels=["Low", "Medium", "High"]
)

print("\nCustomer Segment Distribution:")
print(customer["Segment"].value_counts())

def make_three_categories(series, labels):
    ranks = series.rank(method="first")
    return pd.qcut(ranks, q=3, labels=labels)

customer["Recency_Category"] = make_three_categories(customer["Recency"], ["Recent", "Moderate", "Inactive"])
customer["Frequency_Category"] = make_three_categories(customer["Frequency"], ["Low", "Medium", "High"])
customer["Quantity_Category"] = make_three_categories(customer["TotalQuantity"], ["Low", "Medium", "High"])
customer["Product_Category"] = make_three_categories(customer["UniqueProducts"], ["Few", "Moderate", "Many"])

features = [
    "Recency_Category",
    "Frequency_Category",
    "Quantity_Category",
    "Product_Category"
]

X = customer[features].astype(str)
y = customer["Segment"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

def entropy(y):
    if len(y) == 0:
        return 0

    values, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)

    result = 0

    for p in probabilities:
        if p > 0:
            result -= p * np.log2(p)

    return result

def information_gain(X, y, feature):
    parent_entropy = entropy(y)
    values, counts = np.unique(X[feature], return_counts=True)
    weighted_entropy = 0

    for value, count in zip(values, counts):
        mask = X[feature] == value
        y_subset = y[mask]
        weight = len(y_subset) / len(y)
        weighted_entropy += weight * entropy(y_subset)

    return parent_entropy - weighted_entropy

print("\nInformation Gain:")

for feature in features:
    gain = information_gain(X_train, y_train, feature)
    print(f"{feature}: {gain:.6f}")

class Node:
    def __init__(self, feature=None, prediction=None, gain=0):
        self.feature = feature
        self.prediction = prediction
        self.gain = gain
        self.children = {}

def majority_class(y):
    return y.value_counts().idxmax()

def build_tree(X, y, available_features, depth=0, max_depth=4):
    if len(y.unique()) == 1:
        return Node(prediction=y.iloc[0])

    if len(available_features) == 0:
        return Node(prediction=majority_class(y))

    if depth >= max_depth:
        return Node(prediction=majority_class(y))

    gains = {}

    for feature in available_features:
        gains[feature] = information_gain(X, y, feature)

    best_feature = max(gains, key=gains.get)
    best_gain = gains[best_feature]

    if best_gain <= 0:
        return Node(prediction=majority_class(y))

    node = Node(feature=best_feature, gain=best_gain)

    remaining_features = [feature for feature in available_features if feature != best_feature]

    for value in X[best_feature].unique():
        mask = X[best_feature] == value
        X_subset = X[mask]
        y_subset = y[mask]

        if len(y_subset) == 0:
            node.children[value] = Node(prediction=majority_class(y))
        else:
            node.children[value] = build_tree(X_subset, y_subset, remaining_features, depth + 1, max_depth)

    return node

tree = build_tree(X_train, y_train, features, max_depth=4)

def print_tree(node, depth=0, branch="ROOT"):
    indent = "    " * depth

    if node.prediction is not None:
        print(indent + f"{branch} --> Predict: {node.prediction}")
        return

    print(indent + f"{branch} --> {node.feature} (Gain={node.gain:.4f})")

    for value, child in node.children.items():
        print(indent + f"    |-- {value}")
        print_tree(child, depth + 2)

print("\nManual ID3 Decision Tree:")
print_tree(tree)

def predict_one(node, sample, fallback):
    if node.prediction is not None:
        return node.prediction

    value = sample[node.feature]

    if value in node.children:
        return predict_one(node.children[value], sample, fallback)

    return fallback

fallback_class = majority_class(y_train)
predictions = []

for _, row in X_test.iterrows():
    predictions.append(predict_one(tree, row, fallback_class))

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, predictions))

feature_importance = {feature: 0.0 for feature in features}

def calculate_importance(node, X, y):
    if node.prediction is not None:
        return

    feature = node.feature
    feature_importance[feature] += node.gain * len(y)

    for value, child in node.children.items():
        mask = X[feature] == value
        X_child = X[mask]
        y_child = y[mask]

        if len(y_child) > 0:
            calculate_importance(child, X_child, y_child)

calculate_importance(tree, X_train, y_train)

total_importance = sum(feature_importance.values())

if total_importance > 0:
    for feature in feature_importance:
        feature_importance[feature] /= total_importance

sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

print("\nFeature Importance:")

for feature, importance in sorted_importance:
    print(f"{feature}: {importance:.4f}")

importance_features = [x[0] for x in sorted_importance]
importance_values = [x[1] for x in sorted_importance]

plt.figure(figsize=(9, 5))
plt.bar(importance_features, importance_values)
plt.xlabel("Features")
plt.ylabel("Normalized Importance")
plt.title("Feature Importance - Manual ID3")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

def tree_depth(node):
    if node.prediction is not None:
        return 1

    return 1 + max(tree_depth(child) for child in node.children.values())

def count_nodes(node):
    if node.prediction is not None:
        return 1

    return 1 + sum(count_nodes(child) for child in node.children.values())

print("\nTree Depth:", tree_depth(tree))
print("Total Nodes:", count_nodes(tree))

def assign_positions(node, depth=0, positions=None, counter=None):
    if positions is None:
        positions = {}

    if counter is None:
        counter = [0]

    if node.prediction is not None:
        positions[id(node)] = (counter[0], -depth)
        counter[0] += 1
        return positions

    for child in node.children.values():
        assign_positions(child, depth + 1, positions, counter)

    child_positions = [positions[id(child)] for child in node.children.values()]
    min_x = min(x for x, y in child_positions)
    max_x = max(x for x, y in child_positions)

    positions[id(node)] = ((min_x + max_x) / 2, -depth)

    return positions

positions = assign_positions(tree)

def draw_tree(node, positions, ax):
    x, y = positions[id(node)]

    if node.prediction is not None:
        text = f"Predict:\n{node.prediction}"
    else:
        text = f"{node.feature}\nGain={node.gain:.3f}"

    ax.text(
        x, y, text,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", edgecolor="black")
    )

    if node.prediction is None:
        for value, child in node.children.items():
            child_x, child_y = positions[id(child)]

            ax.plot([x, child_x], [y, child_y], "k-")

            ax.text(
                (x + child_x) / 2,
                (y + child_y) / 2,
                str(value),
                ha="center",
                va="center",
                fontsize=9
            )

            draw_tree(child, positions, ax)

plt.figure(figsize=(18, 10))
ax = plt.gca()

draw_tree(tree, positions, ax)

ax.axis("off")
plt.title("Manual ID3 Decision Tree - Online Retail")
plt.tight_layout()
plt.show()
import csv
import random
import math
from collections import Counter

# Load Dataset
def load_dataset(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        data = [row for row in reader]
    return header, data


# Handle Missing Values
def handle_missing(data):
    for col in range(len(data[0])):
        column_values = [row[col] for row in data if row[col] != ""]
        if not column_values:
            continue
        most_common = Counter(column_values).most_common(1)[0][0]
        for row in data:
            if row[col] == "":
                row[col] = most_common
    return data


# Encode Categorical Values
def encode_data(data):
    encoders = {}
    for col in range(len(data[0])):
        unique_vals = list(set(row[col] for row in data))
        encoders[col] = {val: i for i, val in enumerate(unique_vals)}
        for row in data:
            row[col] = encoders[col][row[col]]
    return data


# Train/Test Split
def train_test_split(data, test_size=0.3):
    random.shuffle(data)
    split_index = int(len(data) * (1 - test_size))
    return data[:split_index], data[split_index:]


# Gini Index
def gini_index(groups, classes):
    total_samples = sum(len(group) for group in groups)
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        labels = [row[-1] for row in group]
        for class_val in classes:
            proportion = labels.count(class_val) / size
            score += proportion ** 2
        gini += (1 - score) * (size / total_samples)
    return gini


# Split Dataset
def test_split(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Find Best Split
def get_best_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    best_index, best_value, best_score, best_groups = None, None, float("inf"), None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < best_score:
                best_index, best_value, best_score, best_groups = index, row[index], gini, groups
    return {"index": best_index, "value": best_value, "groups": best_groups}


# Create Leaf Node
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# Build Decision Tree
def split(node, max_depth, min_size, depth):
    left, right = node["groups"]
    del(node["groups"])

    if not left or not right:
        node["left"] = node["right"] = to_terminal(left + right)
        return

    if depth >= max_depth:
        node["left"], node["right"] = to_terminal(left), to_terminal(right)
        return

    if len(left) <= min_size:
        node["left"] = to_terminal(left)
    else:
        node["left"] = get_best_split(left)
        split(node["left"], max_depth, min_size, depth + 1)

    if len(right) <= min_size:
        node["right"] = to_terminal(right)
    else:
        node["right"] = get_best_split(right)
        split(node["right"], max_depth, min_size, depth + 1)


def build_tree(train, max_depth, min_size):
    root = get_best_split(train)
    split(root, max_depth, min_size, 1)
    return root


# Prediction
def predict(node, row):
    if row[node["index"]] < node["value"]:
        if isinstance(node["left"], dict):
            return predict(node["left"], row)
        else:
            return node["left"]
    else:
        if isinstance(node["right"], dict):
            return predict(node["right"], row)
        else:
            return node["right"]


# Random Forest
def subsample(dataset, ratio):
    sample = []
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        sample.append(random.choice(dataset))
    return sample


def random_forest(train, test, n_trees=5, max_depth=5, min_size=2, sample_size=0.8):
    trees = []
    for _ in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size)
        trees.append(tree)

    predictions = []
    for row in test:
        tree_preds = [predict(tree, row) for tree in trees]
        final_pred = max(set(tree_preds), key=tree_preds.count)
        predictions.append(final_pred)
    return predictions


# Accuracy & Classification Report
def accuracy_score(actual, predicted):
    correct = sum(1 for a, p in zip(actual, predicted) if a == p)
    return correct / len(actual)


def classification_report(actual, predicted):
    classes = set(actual)
    for cls in classes:
        tp = sum((a == cls and p == cls) for a, p in zip(actual, predicted))
        fp = sum((a != cls and p == cls) for a, p in zip(actual, predicted))
        fn = sum((a == cls and p != cls) for a, p in zip(actual, predicted))

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

        print(f"\nClass {cls}")
        print("Precision:", round(precision, 3))
        print("Recall:", round(recall, 3))
        print("F1-Score:", round(f1, 3))


# MAIN EXECUTION

print("Loading dataset...")
header, data = load_dataset("webLogin_Intrusion_Dataset.csv")

print("Handling missing values...")
data = handle_missing(data)

print("Encoding categorical data...")
data = encode_data(data)

data = [list(map(int, row)) for row in data]

train, test = train_test_split(data, test_size=0.3)

print("Training Random Forest...")
predictions = random_forest(train, test)

actual = [row[-1] for row in test]

acc = accuracy_score(actual, predictions)

print("\nAccuracy:", round(acc, 4))
print("\nClassification Report:")
classification_report(actual, predictions)


# Lab 04 - Activity 1
# KNN for Credit Card Fraud Detection

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Step 1: Load the dataset
print("Loading dataset...")

data = pd.read_csv("creditcard.csv")

print("\nDataset Preview:")
print(data.head())

print("\nDataset Shape:", data.shape)

# Step 2: Check class distribution
print("\nClass Distribution:")
print(data["Class"].value_counts())

# Step 3: Select a stratified sample to reduce processing time
# Stratification preserves the original fraud/non-fraud ratio.

# Step 3: Select a stratified sample
data = pd.concat([
    data[data["Class"] == 0].sample(
        n=50000, random_state=42
    ),
    data[data["Class"] == 1]
], ignore_index=True)

print("\nSampled Dataset Shape:", data.shape)
print("\nSampled Class Distribution:")
print(data["Class"].value_counts())


# Step 4: Separate features and target
X = data.drop("Class", axis=1)
y = data["Class"]

# Step 5: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# Step 6: Build a pipeline
# Scaling and SMOTE are fitted only on training folds.
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("knn", KNeighborsClassifier())
])

# Step 7: Train KNN with k=5
print("\nTraining KNN with k=5...")

pipeline.set_params(knn__n_neighbors=5)
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n--- KNN Results (k=5) ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")

# Step 8: Tune k using GridSearchCV
print("\nFinding the best value of k...")

param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=3,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters:", grid_search.best_params_)
print(f"Best Cross-Validation F1-Score: {grid_search.best_score_:.4f}")

# Step 9: Evaluate the best model on test data
best_model = grid_search.best_estimator_
best_predictions = best_model.predict(X_test)

best_accuracy = accuracy_score(y_test, best_predictions)
best_f1 = f1_score(y_test, best_predictions, zero_division=0)

print("\n--- Tuned KNN Results ---")
print(f"Best k: {grid_search.best_params_['knn__n_neighbors']}")
print(f"Accuracy: {best_accuracy:.4f}")
print(f"F1-Score: {best_f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    best_predictions,
    target_names=["Normal Transaction", "Fraud Transaction"],
    zero_division=0
))

# Step 10: Compare the results
print("\n--- Performance Comparison ---")
print(f"{'Model':<25}{'Accuracy':<15}{'F1-Score':<15}")
print("-" * 55)
print(f"{'KNN (k=5)':<25}{accuracy:<15.4f}{f1:<15.4f}")
print(f"{'Tuned KNN':<25}{best_accuracy:<15.4f}{best_f1:<15.4f}")

print("\nActivity 1 Completed Successfully!")

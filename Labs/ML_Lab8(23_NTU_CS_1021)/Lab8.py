# Machine Learning Lab 08 - Slides 28 to 38

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_predict, cross_val_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE


# SETUP: Load and prepare the Titanic dataset
# These preparation steps are prerequisites from earlier slides.

try:
    raw_data = pd.read_csv("titanic.csv")
except FileNotFoundError:
    raise FileNotFoundError(
        "titanic.csv was not found. Put the Titanic CSV file in the same "
        "folder as this Python script, then run it again."
    )


if "Gender" not in raw_data.columns and "Sex" in raw_data.columns:
    raw_data = raw_data.rename(columns={"Sex": "Gender"})

feature_columns = ["Pclass", "Gender", "Age", "SibSp", "Parch", "Fare"]
required_columns = ["Survived"] + feature_columns
missing_columns = [c for c in required_columns if c not in raw_data.columns]
if missing_columns:
    raise ValueError(
        f"Dataset is missing required columns: {missing_columns}. "
        "Expected Survived, Pclass, Gender/Sex, Age, SibSp, Parch, Fare."
    )

data = raw_data[required_columns].copy()


encoder = LabelEncoder()
data["Gender"] = encoder.fit_transform(data["Gender"].astype(str))

age_imputer = SimpleImputer(strategy="median")
data[["Age"]] = age_imputer.fit_transform(data[["Age"]])

X = data.drop("Survived", axis=1)
y = data["Survived"].astype(int)

print("Dataset shape:", data.shape)
print("Missing values after preprocessing:")
print(data.isna().sum())


# PREREQUISITE from the earlier Train/Validation/Test example:
# determine best_depth for the final Random Forest model.

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

depth_values = [2, 4, 6, 8, None]
depth_results = []
for depth in depth_values:
    candidate = RandomForestClassifier(max_depth=depth, random_state=42)
    candidate.fit(X_train, y_train)
    train_accuracy = accuracy_score(y_train, candidate.predict(X_train))
    val_accuracy = accuracy_score(y_val, candidate.predict(X_val))
    depth_results.append([depth, train_accuracy, val_accuracy])

depth_table = pd.DataFrame(
    depth_results,
    columns=["Max Depth", "Training Accuracy", "Validation Accuracy"]
)
best_depth = depth_table.loc[
    depth_table["Validation Accuracy"].idxmax(), "Max Depth"
]
if pd.isna(best_depth):
    best_depth = None

print("\nBest Random Forest depth (prerequisite):", best_depth)


# PREREQUISITE from earlier Cross-Validation example:
# generate cross-validated probabilities used in Slide 28.

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
probability_model = RandomForestClassifier(random_state=42)
probabilities = cross_val_predict(
    probability_model, X, y, cv=folds, method="predict_proba"
)[:, 1]


# SLIDE 28 - E. Classification Threshold Comparison

threshold_results = []

for threshold in np.arange(0.1, 1.0, 0.1):
    threshold_predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, threshold_predictions).ravel()

    threshold_results.append({
        "Threshold": round(threshold, 2),
        "Accuracy": accuracy_score(y, threshold_predictions),
        "Precision": precision_score(y, threshold_predictions, zero_division=0),
        "Recall": recall_score(y, threshold_predictions, zero_division=0),
        "F1-score": f1_score(y, threshold_predictions, zero_division=0),
        "Specificity": tn / (tn + fp)
    })

threshold_table = pd.DataFrame(threshold_results)
print("\nSLIDE 28 - Classification Threshold Comparison:")
print(threshold_table.round(6).to_string(index=False))

# Find the threshold with the highest F1-score
best_row = threshold_table.loc[threshold_table["F1-score"].idxmax()]
print("Best threshold:", best_row["Threshold"])
print("Best F1-score:", round(best_row["F1-score"], 3))
threshold_table.to_csv("slide28_threshold_results.csv", index=False)

# SLIDES 30-31 - F. Overfitting and Underfitting

depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, None]
fit_results = []

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for depth in depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)

  
    tree.fit(X, y)
    train_accuracy = accuracy_score(y, tree.predict(X))


    cv_accuracy = cross_val_score(tree, X, y, cv=folds).mean()

    fit_results.append({
        "Max Depth": str(depth),
        "Training Accuracy": train_accuracy,
        "Cross-validation Accuracy": cv_accuracy
    })

fit_table = pd.DataFrame(fit_results)
print("\nSLIDE 30 - Overfitting and Underfitting results:")
print(fit_table.round(6).to_string(index=False))
fit_table.to_csv("slides30_31_overfitting_results.csv", index=False)

# Slide 31 graph
plt.figure(figsize=(8, 5))
plt.plot(
    fit_table["Max Depth"], fit_table["Training Accuracy"],
    marker="o", label="Training"
)
plt.plot(
    fit_table["Max Depth"], fit_table["Cross-validation Accuracy"],
    marker="o", label="Cross-validation"
)
plt.xlabel("Decision Tree Maximum Depth")
plt.ylabel("Accuracy")
plt.title("Overfitting and Underfitting")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("slides30_31_overfitting.png")
plt.show()


# SLIDE 34 - G. Class Imbalance

# Keep all people who died and randomly select 47 survivors
died = data[data["Survived"] == 0]
survivors = data[data["Survived"] == 1].sample(47, random_state=42)

imbalanced_data = pd.concat([died, survivors]).sample(frac=1, random_state=42)

X_imb = imbalanced_data.drop("Survived", axis=1)
y_imb = imbalanced_data["Survived"]

print("\nSLIDE 34 - Class Imbalance counts:")
print(y_imb.value_counts())
print("Survivor percentage:", round(y_imb.mean() * 100, 2), "%")


# SLIDE 35 - G. SMOTE Demonstration

# Select one training fold
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_index, test_index = next(folds.split(X_imb, y_imb))

X_train_smote = X_imb.iloc[train_index]
y_train_smote = y_imb.iloc[train_index]

print("\nSLIDE 35 - Before SMOTE:")
print(y_train_smote.value_counts())

# Apply SMOTE to the training data only
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train_smote, y_train_smote
)

print("\nAfter SMOTE:")
print(pd.Series(y_train_balanced).value_counts())

# SLIDE 37 - H. Train and Save the Final Model

# Train the final model using all available labelled data
final_model = RandomForestClassifier(
    max_depth=best_depth,
    n_estimators=100,
    random_state=42
)
final_model.fit(X, y)

# Save the model and preprocessing information together
model_bundle = {
    "model": final_model,
    "gender_encoder": encoder,
    "age_imputer": age_imputer,
    "feature_columns": list(X.columns),
    "target_name": "Survived"
}

joblib.dump(model_bundle, "titanic_random_forest_model.joblib")
print("\nFinal model saved as titanic_random_forest_model.joblib")

# Example: load the saved model and make a prediction
loaded_bundle = joblib.load("titanic_random_forest_model.joblib")
loaded_model = loaded_bundle["model"]
print(
    "Loaded model prediction for the first passenger:",
    loaded_model.predict(X.iloc[[0]])
)


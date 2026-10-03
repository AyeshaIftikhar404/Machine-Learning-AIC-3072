import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)


# ==========================================
# LOAD DATASET
# ==========================================

data = pd.read_csv("Social_Network_Ads.csv")

print("Dataset Loaded Successfully!")
print(data.head())


# ==========================================
# PREPROCESSING
# ==========================================

# Convert Gender into numeric values
data["Gender"] = data["Gender"].map({
    "Female": 0,
    "Male": 1
})


# Features and Target
X = data[["Gender", "Age", "EstimatedSalary"]]
y = data["Purchased"]


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==========================================
# LOGISTIC REGRESSION
# ==========================================

model = LogisticRegression()

model.fit(X_train, y_train)


# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# Probability of class 1
y_prob = model.predict_proba(X_test)[:, 1]


# ==========================================
# EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print("\n===================================")
print("LOGISTIC REGRESSION RESULTS")
print("===================================")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-Score :", round(f1, 4))


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()
plt.title("Confusion Matrix")
plt.show()


# ==========================================
# ROC CURVE
# ==========================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

auc_score = roc_auc_score(
    y_test,
    y_prob
)

print("\nROC-AUC Score:", round(auc_score, 4))


plt.plot(
    fpr,
    tpr,
    label="ROC Curve"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()
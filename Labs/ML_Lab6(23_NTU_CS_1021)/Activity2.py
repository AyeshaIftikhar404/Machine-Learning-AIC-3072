import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]
data = pd.read_csv(url, names=columns)

print("Dataset shape:", data.shape)
print("First 10 rows:")
print(data.head(10))


data.replace('?', pd.NA, inplace=True)
data.dropna(inplace=True)
data = data.astype(float)

data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)

print("Missing values after cleaning:\n", data.isnull().sum())
print("Target class distribution:\n", data['target'].value_counts())


X = data.drop(columns=['target'])
y = data['target']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


base_rf = RandomForestClassifier(random_state=42)
base_rf.fit(X_train, y_train)
base_pred = base_rf.predict(X_test)
print("\nBasic Model Test Accuracy:", accuracy_score(y_test, base_pred))

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nBest Hyperparameters:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n--- Test Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


importances = best_model.feature_importances_
feat_importances = pd.Series(importances, index=X.columns)
plt.figure(figsize=(8, 5))
feat_importances.nlargest(3).plot(kind='barh', color='teal')
plt.title('Top 3 Most Influential Features')
plt.xlabel('Relative Importance')
plt.show()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE

# 1. Dataset Load karein (Ensure karein ke creditcard.csv same folder mein ho)
print("Loading dataset...")
df = pd.read_csv('creditcard.csv').sample(n=10000, random_state=42)

# Features (X) aur Target (y) separate karein
X = df.drop('Class', axis=1)
y = df['Class']

# 2. Train-Test Split (stratify=y taake class balance maintain rahe)
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. SMOTE ka use karke class imbalance handle karein
print("Applying SMOTE for handling class imbalance...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 4. StandardScaler se features scale karein
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# 5. KNN Classifier train karein aur GridSearchCV se 'k' tune karein
print("Training KNN and tuning with GridSearchCV...")
param_grid = {'n_neighbors': [3, 5]}  # Time bachane ke liye chota grid rakha hai
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=3, scoring='f1', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train_res)

best_knn = grid_search.best_estimator_
print("Best Parameters found:", grid_search.best_params_)

# 6. Model Evaluation
print("Evaluating model...")
y_pred = best_knn.predict(X_test_scaled)

print("\n--- ACTIVITY 1 RESULTS ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
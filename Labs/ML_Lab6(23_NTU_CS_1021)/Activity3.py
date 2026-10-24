import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    data = pd.read_csv('train.csv')
except FileNotFoundError:
    print("plz download file.")


if 'data' in locals():
    print("Dataset shape:", data.shape)
    print("First 10 rows:\n", data.head(10))
    print("Summary statistics of SalePrice:\n", data['SalePrice'].describe())


    if 'Id' in data.columns:
        data = data.drop(columns=['Id'])
    
    X = data.drop(columns=['SalePrice'])
    y = data['SalePrice']

    
    X = pd.get_dummies(X)
    X.fillna(X.mean(), inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    default_rf = RandomForestRegressor(random_state=42)
    default_rf.fit(X_train, y_train)
    def_pred = default_rf.predict(X_test)
    print("\nDefault Model R2 Score:", r2_score(y_test, def_pred))


    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': [0.7, 1.0]
    }

    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("\nBest Hyperparameters:", grid_search.best_params_)
    best_cv_mse = -grid_search.best_score_
    print("Best CV RMSE:", np.sqrt(best_cv_mse))


    best_regressor = grid_search.best_estimator_
    y_pred = best_regressor.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Test Evaluation ---")
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='purple')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual SalePrice')
    plt.ylabel('Predicted SalePrice')
    plt.title('Actual vs Predicted House Prices')
    plt.grid(alpha=0.3)
    plt.show()

    importances = best_regressor.feature_importances_
    feat_importances = pd.Series(importances, index=X.columns)
    plt.figure(figsize=(8, 5))
    feat_importances.nlargest(5).plot(kind='barh', color='coral')
    plt.title('Top 5 Most Influential Features')
    plt.xlabel('Relative Importance')
    plt.show()
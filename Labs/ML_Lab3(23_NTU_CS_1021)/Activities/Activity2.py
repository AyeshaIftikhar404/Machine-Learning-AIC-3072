import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score


# ==========================================
# LOAD DATASET
# ==========================================

data = pd.read_csv("Medical Cost Personal Datasets.csv")

print("Dataset Loaded Successfully!")
print(data.head())


# ==========================================
# FEATURES AND TARGET
# ==========================================

# Independent variable
X = data[["age"]]

# Dependent variable
y = data["charges"]


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
# LINEAR REGRESSION
# ==========================================

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_pred = linear_model.predict(
    X_test
)

linear_r2 = r2_score(
    y_test,
    linear_pred
)


print("\n===================================")
print("LINEAR REGRESSION")
print("===================================")

print("R² Score:", round(linear_r2, 4))


# ==========================================
# POLYNOMIAL REGRESSION
# ==========================================

degrees = [2, 3, 4]

polynomial_scores = {}

for degree in degrees:

    # Polynomial transformation
    poly = PolynomialFeatures(
        degree=degree
    )

    X_train_poly = poly.fit_transform(
        X_train
    )

    X_test_poly = poly.transform(
        X_test
    )


    # Create model
    poly_model = LinearRegression()

    poly_model.fit(
        X_train_poly,
        y_train
    )


    # Prediction
    y_pred = poly_model.predict(
        X_test_poly
    )


    # R² Score
    r2 = r2_score(
        y_test,
        y_pred
    )

    polynomial_scores[degree] = r2

    print("\nPolynomial Degree:", degree)
    print("R² Score:", round(r2, 4))


# ==========================================
# COMPARISON
# ==========================================

print("\n===================================")
print("MODEL COMPARISON")
print("===================================")

print(
    "Linear Regression R²:",
    round(linear_r2, 4)
)

for degree, score in polynomial_scores.items():

    print(
        "Polynomial Degree",
        degree,
        "R²:",
        round(score, 4)
    )


# ==========================================
# PLOT
# ==========================================

# Sort values for smooth curves
plot_data = data.sort_values("age")

X_plot = plot_data[["age"]]
y_plot = plot_data["charges"]


# Linear Regression curve
linear_curve = linear_model.predict(
    X_plot
)


plt.figure()

plt.scatter(
    X,
    y,
    label="Actual Data"
)

plt.plot(
    X_plot,
    linear_curve,
    label="Linear Regression"
)

plt.xlabel("Age")
plt.ylabel("Medical Charges")

plt.title("Linear Regression")

plt.legend()

plt.show()


# ==========================================
# POLYNOMIAL CURVES
# ==========================================

plt.figure()

plt.scatter(
    X,
    y,
    label="Actual Data"
)


for degree in degrees:

    poly = PolynomialFeatures(
        degree=degree
    )

    X_poly_all = poly.fit_transform(
        X_plot
    )

    poly_model = LinearRegression()

    poly_model.fit(
        poly.fit_transform(X_train),
        y_train
    )

    curve = poly_model.predict(
        X_poly_all
    )

    plt.plot(
        X_plot,
        curve,
        label=f"Polynomial Degree {degree}"
    )


plt.xlabel("Age")
plt.ylabel("Medical Charges")

plt.title(
    "Polynomial Regression Comparison"
)

plt.legend()

plt.show()
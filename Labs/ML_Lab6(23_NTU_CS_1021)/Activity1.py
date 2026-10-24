import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


print("Dataset load ho raha hai...")
fashion = fetch_openml('Fashion-MNIST', version=1, as_frame=True, parser='auto')
X = fashion.data
y = fashion.target

print("Original dataset shape:", X.shape)
print("First 10 rows:")
print(X.head(10))
print("Class distribution:")
print(y.value_counts())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


pca_50 = PCA(n_components=50)
X_pca_50 = pca_50.fit_transform(X_scaled)


pca_95 = PCA(n_components=0.95)
X_pca_95 = pca_95.fit_transform(X_scaled)

pca_all = PCA(n_components=None)
pca_all.fit(X_scaled)
cumsum = np.cumsum(pca_all.explained_variance_ratio_)

print("\n--- Results Summary ---")
print("Method 1 (50 components) - Explained Variance:", np.sum(pca_50.explained_variance_ratio_))
print("Method 2 (0.95 variance) - Components retained:", pca_95.n_components_)
print("Method 3 (All components) - Total components:", len(pca_all.explained_variance_ratio_))


plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumsum) + 1), cumsum, marker='.', linestyle='-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance Threshold')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance (Fashion MNIST)')
plt.legend()
plt.grid(True)
plt.show()


pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y.astype(int), cmap='tab10', alpha=0.5, s=10)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA 2D Visualization of Fashion MNIST')
plt.colorbar(scatter, label='Clothing Class')
plt.grid(alpha=0.3)
plt.show()
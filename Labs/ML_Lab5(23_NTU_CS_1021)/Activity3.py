import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN

# 1. Load Dataset
df = pd.read_csv('Mall_Customers.csv')
print("Dataset Preview:")
print(df.head())

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), wcss, marker='o', color='red')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Re-run K-Means on Scaled Data
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_init = kmeans.fit_predict(X_scaled)

# 5. Apply DBSCAN Algorithm
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

# Plotting K-Means Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='KMeans_Cluster', data=df, palette='viridis', s=100)
plt.title('K-Means Clusters (Annual Income vs Spending Score)')
plt.show()

# 6. Adding Fake Feature "LuckyNumber" and checking its effect
np.random.seed(42)
df['LuckyNumber'] = np.random.randint(0, 101, size=len(df))

# Re-run K-Means with fake feature
X_with_fake = df[['Annual Income (k$)', 'Spending Score (1-100)', 'LuckyNumber']]
X_with_fake_scaled = scaler.fit_transform(X_with_fake)
kmeans_fake = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster_With_Fake'] = kmeans_fake.fit_predict(X_with_fake_scaled)

print("\n--- Explanation of Fake Feature ---")
print("Jab hum 'LuckyNumber' jaisa random fake feature add karte hain, toh Euclidean distance calculations distorts ho jati hain kyunki random noise clustering boundaries ko kharab kar deta hai aur clusters ki quality kam ho jati hai.")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

data = pd.read_csv("customer_segmentation_dataset.csv")

X = data.drop(columns=["True_Segment"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

data["Cluster"] = clusters

print("Silhouette Score:", silhouette_score(X_scaled, clusters))

pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)

data["PCA1"] = pca_data[:, 0]
data["PCA2"] = pca_data[:, 1]

plt.figure(figsize=(8,6))
sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", palette="Set2", data=data)
plt.title("Customer Segmentation using PCA")
plt.show()

print(data.groupby("Cluster").mean())

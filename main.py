import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv("marketing_campaign.csv", sep="\t")

print("Initial Shape:", df.shape)

df = df.dropna()

df["Total_Spending"] = (
    df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] +
    df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
)

rfm = df[["Recency", "Total_Spending", "NumWebPurchases",
          "NumCatalogPurchases", "NumStorePurchases"]]

rfm["Frequency"] = (
    rfm["NumWebPurchases"] +
    rfm["NumCatalogPurchases"] +
    rfm["NumStorePurchases"]
)

rfm = rfm[["Recency", "Frequency", "Total_Spending"]]

Q1 = rfm.quantile(0.25)
Q3 = rfm.quantile(0.75)
IQR = Q3 - Q1

rfm = rfm[~((rfm < (Q1 - 1.5 * IQR)) |
            (rfm > (Q3 + 1.5 * IQR))).any(axis=1)]

print("Shape after outlier removal:", rfm.shape)

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(rfm_scaled)

kmeans_silhouette = silhouette_score(rfm_scaled, kmeans_labels)
print("K-Means Silhouette Score:", round(kmeans_silhouette, 3))

hierarchical = AgglomerativeClustering(n_clusters=4)
hier_labels = hierarchical.fit_predict(rfm_scaled)

hier_silhouette = silhouette_score(rfm_scaled, hier_labels)
print("Hierarchical Silhouette Score:", round(hier_silhouette, 3))

pca = PCA(n_components=2)
pca_components = pca.fit_transform(rfm_scaled)

plt.figure()
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=kmeans_labels)
plt.title("Customer Segments (K-Means + PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

rfm["Cluster"] = kmeans_labels
cluster_summary = rfm.groupby("Cluster").mean()

print("\nCluster Summary:")
print(cluster_summary)

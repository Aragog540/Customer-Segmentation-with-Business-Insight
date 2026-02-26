import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    return df.dropna()


def engineer_rfm(df: pd.DataFrame) -> pd.DataFrame:
    df["Total_Spending"] = (
        df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] +
        df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
    )

    rfm = df[[
        "Recency",
        "Total_Spending",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases"
    ]].copy()

    rfm["Frequency"] = (
        rfm["NumWebPurchases"] +
        rfm["NumCatalogPurchases"] +
        rfm["NumStorePurchases"]
    )

    return rfm[["Recency", "Frequency", "Total_Spending"]]


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    return df[~((df < (Q1 - 1.5 * IQR)) |
                (df > (Q3 + 1.5 * IQR))).any(axis=1)]


def scale_features(df: pd.DataFrame):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    return scaled, scaler


def run_kmeans(data, n_clusters=4):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(data)
    score = silhouette_score(data, labels)
    return model, labels, score


def run_hierarchical(data, n_clusters=4):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(data)
    score = silhouette_score(data, labels)
    return model, labels, score


def visualize_clusters(data, labels):
    pca = PCA(n_components=2)
    components = pca.fit_transform(data)

    plt.figure()
    plt.scatter(components[:, 0], components[:, 1], c=labels)
    plt.title("Customer Segments (PCA Projection)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()


def cluster_summary(df: pd.DataFrame, labels):
    df_copy = df.copy()
    df_copy["Cluster"] = labels
    return df_copy.groupby("Cluster").mean()


def main():
    df = load_data("marketing_campaign.csv")
    rfm = engineer_rfm(df)
    rfm = remove_outliers(rfm)

    scaled_data, scaler = scale_features(rfm)

    kmeans_model, k_labels, k_score = run_kmeans(scaled_data)
    h_model, h_labels, h_score = run_hierarchical(scaled_data)

    print(f"K-Means Silhouette Score: {round(k_score, 3)}")
    print(f"Hierarchical Silhouette Score: {round(h_score, 3)}")

    visualize_clusters(scaled_data, k_labels)

    summary = cluster_summary(rfm, k_labels)
    print("\nCluster Summary:")
    print(summary)


if __name__ == "__main__":
    main()

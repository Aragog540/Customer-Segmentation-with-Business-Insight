

# Customer Segmentation with Business Insight

## Overview

This project implements an end-to-end customer segmentation pipeline using unsupervised machine learning techniques. The objective is to identify meaningful customer groups based on purchasing behavior and translate them into actionable business insights.

The pipeline includes RFM-based feature engineering, clustering model comparison, and PCA-based visualization.

---

## Dataset

**Dataset Used:** Customer Personality Analysis (Kaggle)

The dataset contains customer demographic details, purchasing history, campaign responses, and transaction behavior.

Key attributes used:

* Recency
* Purchase frequency
* Total spending across product categories
* Web, catalog, and store purchases

---

## Problem Statement

Businesses need to identify high-value customers and churn-risk segments to optimize marketing strategies. This project addresses that challenge by:

* Engineering behavioral features
* Applying clustering algorithms
* Comparing model performance
* Translating clusters into business insights

---

## Project Pipeline

### 1. Data Preprocessing

* Missing value handling
* RFM feature engineering
* Outlier removal using IQR method

### 2. Feature Engineering

* **Recency**
* **Frequency** (combined web, catalog, and store purchases)
* **Monetary Value** (Total Spending)

### 3. Feature Scaling

* StandardScaler normalization

### 4. Clustering

* K-Means
* Hierarchical (Agglomerative) Clustering
* Model comparison using Silhouette Score

### 5. Dimensionality Reduction

* PCA for 2D cluster visualization

### 6. Business Insight Extraction

* Cluster profiling using group-wise averages
* Identification of premium, moderate, and churn-risk segments

---

## Model Evaluation

Models were evaluated using:

* Silhouette Score
* Cluster separation comparison
* Interpretability of segments

K-Means demonstrated stronger separation performance after feature optimization.

---

## Business Insights

Example outcomes from clustering:

* High-frequency, high-spending customers identified as premium segment
* Low-frequency, recent customers flagged as churn-risk
* Mid-tier customers suitable for targeted upselling campaigns

These insights can support:

* Targeted marketing
* Customer retention strategies
* Campaign personalization

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* SciPy

---

## Project Structure

```
├── marketing_campaign.csv
├── customer_segmentation.py
├── requirements.txt
└── README.md
```

---

## How to Run

Install dependencies:

```
pip install -r requirements.txt
```

Run the pipeline:

```
python customer_segmentation.py
```

---

## Future Improvements

* Automatic optimal cluster detection
* DBSCAN comparison
* Model persistence using joblib
* API deployment
* Interactive dashboard integration

---

## Author

Swaroop Bhowmik
B.Tech CSE (AI & ML)

---

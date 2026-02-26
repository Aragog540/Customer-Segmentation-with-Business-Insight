# Customer Segmentation with Business Insight

## 📌 Project Overview

This project applies unsupervised machine learning techniques to segment customers based on demographic and behavioral attributes. Using clustering and dimensionality reduction, it identifies distinct customer groups and generates actionable business insights for targeted marketing and strategic decision-making.

---

## 🎯 Objectives

* Clean and preprocess customer data
* Perform feature scaling
* Apply K-Means clustering
* Evaluate clustering performance
* Reduce dimensionality using PCA
* Visualize customer clusters
* Derive business insights

---

## 📊 Dataset Description

The dataset contains **600 retail customers** with the following features:

* **Age** – Customer age
* **Annual_Income_k$** – Annual income (in thousand dollars)
* **Spending_Score** – Score based on purchasing behavior (1–100)
* **Online_Purchases_per_Month** – Monthly online purchase frequency
* **Tenure_Years** – Years as a customer
* **True_Segment** – Ground-truth label (for validation purposes only)

---

## 🛠 Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---

## ⚙ Methodology

### 1️⃣ Data Preprocessing

* Loaded dataset
* Removed `True_Segment` column for clustering
* Checked for missing values

### 2️⃣ Feature Scaling

Standardized features using `StandardScaler` to ensure equal contribution in distance-based clustering.

### 3️⃣ Clustering

* Applied **K-Means algorithm**
* Used **Elbow Method** to determine optimal cluster count
* Evaluated using **Silhouette Score**

### 4️⃣ Dimensionality Reduction

Used **Principal Component Analysis (PCA)** to reduce dimensions to 2 for visualization.

### 5️⃣ Visualization

Plotted clusters using PCA components to observe separation patterns.

---

## 📂 Project Structure

```
customer_segmentation/
│
├── customer_segmentation_dataset.csv
├── segmentation.py
├── README.md
└── requirements.txt
```

---

## ▶ How to Run

### Install Dependencies

```
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run the Script

```
python segmentation.py
```

---

## 📈 Expected Outputs

* Elbow Curve Plot
* Silhouette Score
* PCA-Based Cluster Visualization
* Cluster-wise Statistical Summary

---

## 💡 Business Insights

The model typically identifies segments such as:

* **Premium Loyal Customers**
* **Impulse Young Buyers**
* **Wealthy Conservative Buyers**
* **Price Sensitive Customers**

These segments help businesses to:

* Design targeted marketing campaigns
* Improve retention strategies
* Personalize recommendations
* Optimize promotional spending

---

## 🚀 Future Improvements

* Implement DBSCAN or Gaussian Mixture Models
* Apply RFM (Recency, Frequency, Monetary) analysis
* Deploy as a Streamlit dashboard
* Use real-world transactional datasets

---

## 👨‍💻 Author

**Swaroop Bhowmik**
B.Tech CSE (AI & ML Specialization)

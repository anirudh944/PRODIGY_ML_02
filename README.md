# PRODIGY_ML_02

# 🛍️ Customer Segmentation using K-Means Clustering

## 📌 Project Overview
This project applies **K-Means clustering** to segment customers based on:
- **Annual Income (k$)**
- **Spending Score (1–100)**

The goal is to group mall customers into meaningful segments for targeted marketing strategies.

---

## 📊 Dataset
- File: **Mall_Customers.csv**
- Columns used for clustering:
  - `Annual Income (k$)`
  - `Spending Score (1-100)`

---

## ⚙️ Technologies Used
- Python  
- Pandas & NumPy (data handling)  
- Scikit-learn (KMeans, StandardScaler)  
- Matplotlib & Seaborn (visualization)  


## 📈 Results
🔹 Elbow Method
- The elbow plot indicated that 5 clusters is optimal.

🔹 Clustered Data (Sample):
CustomerID	Annual Income (k$)	Spending Score (1-100)	Cluster
1	                 15	                  39	               4
2	                 15	                  81	               3
3	                 16	                  6                  4
4	                 16	                  77	               3
5	                 17	                  40                 4
6	                 17	                  76	               3
7	                 18	                  6	                 4
8	                 18	                  94	               1
9	                 19	                  3	                 4
10	               19	                  72	               3


## Visualization:-

Customers were divided into 5 clusters:

Cluster 0: Medium income, medium spending

Cluster 1: High income, high spending (target customers 💰)

Cluster 2: Low income, low spending

Cluster 3: Low income, high spending (value seekers)

Cluster 4: High income, low spending (savers)

Scatter plot clearly shows distinct groups.

## Conclusion:-

K-Means successfully segmented mall customers into 5 distinct groups.

Businesses can now design personalized marketing campaigns:

Reward high spenders with loyalty programs.

Encourage high-income low spenders with premium offers.

Convert value seekers with discounts.



---

## 📜 Code
```python
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("Mall_Customers.csv")

# Select relevant features
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal clusters (Elbow Method)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

# Apply KMeans with 5 clusters (from elbow curve)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster column
data["Cluster"] = clusters

# Visualize clusters
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=data["Annual Income (k$)"],
    y=data["Spending Score (1-100)"],
    hue=data["Cluster"],
    palette="Set2",
    s=80
)
plt.title("Customer Segmentation (K-Means)")
plt.show()

# Show first 10 clustered customers
print(data.head(10))

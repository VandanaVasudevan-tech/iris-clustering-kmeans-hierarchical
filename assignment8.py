from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd


iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")
df = pd.concat([X, y], axis=1)
print(df.head(150))
print(df.isnull().sum())

# KMeans uses Euclidean distance, so features must be on the same scale.
# Step-2
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======================================================================================================================
# KMeans Clustering
# ======================================================================================================================
# Working: Algorithm begin by randomly selecting k cluster centroids.
#          Each data point is assigned to the nearest centroid, forming clusters.
#          After the assignment, It will recalculate the centroid of each cluster by averaging the points within it.
#          This process repeats until the centroids no longer change or the maximum number of iterations is reached.
#          Uses the elbow method to choose the optimal value of k in K-Means
# WHY SUITABLE ?
# K-Means is suitable for the Iris dataset because it contains numerical, low-dimensional data with natural
# and relatively well-separated clusters, making distance-based clustering effective.

# Visualization: Checking which two features shows clear groups and low overlap
# ------------------------------------------------------------------------------

plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], color='red')
plt.scatter(df['petal length (cm)'], df['petal width (cm)'], color='green')
plt.show()

# Step 1: Decide the number of cluster(k) using elbow method
# ------------------------------------------------------------
sse = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    sse.append(km.inertia_)

plt.plot(range(1, 11), sse, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("W-CSS (Inertia)")
plt.title("Elbow Method for Optimal K")
plt.show()

# Step 2: Fit the KMeans model
# -------------------------------------

km = KMeans(n_clusters=4, random_state=42)
km.fit(X_scaled)
labels = km.labels_
df['Cluster'] = labels

# Step 3: visualize the clusters
# ------------------------------------
plt.scatter(df['petal length (cm)'], df['petal width (cm)'], c=df['Cluster'])
plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")
plt.show()
# ======================================================================================================================
# Hierarchical Clustering
# ======================================================================================================================
# WORKING: Hierarchical clustering is an unsupervised learning algorithm that builds a hierarchy of clusters without
# requiring the number of clusters in advance.
# There are two main approaches:
# Agglomerative (bottom-up) – each data point starts as its own cluster, and the closest clusters are repeatedly merged.
# Divisive (top-down) – all data points start in one cluster and are recursively split.
# The clustering process is visualized using a dendrogram, which shows how clusters are merged at different distance
# levels.

# WHY SUITABLE?
# The dataset is small (150 samples), making hierarchical methods computationally feasible.
# Features are continuous numerical values, which work well with distance-based clustering.
# The dataset naturally contains three distinct groups, which can be observed clearly in the dendrogram.
# It does not require pre-specifying the number of clusters, unlike KMeans.
# ======================================================================================================================
# Step-1: Perform Bottom-Up approach Hierarchical clustering (for dendrogram)
linked = linkage(X_scaled, method='ward')

# Step-3: Plot the dendrogram
plt.figure(figsize=(8, 5))
# Shows only top 10 merged clusters
dendrogram(linked, truncate_mode='lastp', p=10)
plt.title('Dendrogram')
plt.xlabel("Data points")
plt.ylabel("Distance")
plt.show()

# Step-4: Apply Bottom-Up approach using sklearn
model = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'
)
labels = model.fit_predict(X_scaled)

# Step-5: Visualize the final clusters
plt.figure(figsize=(8, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
plt.xlabel("sepal Length (scaled)")
plt.ylabel("sepal Width (scaled)")
plt.title("Hierarchical Clustering (Bottom-Up approach) – Iris Dataset")
plt.show()

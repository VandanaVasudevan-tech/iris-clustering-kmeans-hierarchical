# iris-clustering-kmeans-hierarchical
Implementation of KMeans and Hierarchical clustering on the Iris dataset using scikit-learn, including preprocessing, visualization, and algorithm explanations.

📌 Objective

The objective of this assignment is to understand and apply clustering techniques on a real-world dataset using unsupervised machine learning algorithms.
This project implements KMeans Clustering and Hierarchical (Agglomerative) Clustering on the Iris dataset.

📊 Dataset

Dataset: Iris dataset

Source: sklearn.datasets.load_iris

Samples: 150

Features:

Sepal length (cm)

Sepal width (cm)

Petal length (cm)

Petal width (cm)

Since clustering is an unsupervised learning task, the species column is removed before applying the algorithms.

🛠️ Libraries Used

pandas

matplotlib

scikit-learn

scipy

🔄 Data Loading & Preprocessing

Loaded the Iris dataset using sklearn

Converted the data into a Pandas DataFrame

Checked for missing values (none found)

Applied StandardScaler to normalize the features
(Scaling is required as distance-based algorithms like KMeans and Hierarchical clustering rely on Euclidean distance)

🔹 KMeans Clustering
🔍 How KMeans Works

Randomly initializes K cluster centroids

Assigns each data point to the nearest centroid

Recalculates centroids based on cluster means

Repeats until centroids stabilize or maximum iterations are reached

✅ Why KMeans is Suitable for the Iris Dataset

The dataset contains numerical and low-dimensional features

Data points form naturally separable groups

Distance-based clustering works effectively

Computationally efficient for small datasets

📐 Choosing Optimal K (Elbow Method)

The Elbow Method is used to determine the optimal number of clusters

Inertia (Within-Cluster Sum of Squares) is plotted against different K values

The elbow point indicates the best value of K

<img width="786" height="633" alt="image" src="https://github.com/user-attachments/assets/e033e5ee-4208-46e2-ba47-c4e1bf8ac78f" />


<img width="792" height="632" alt="image" src="https://github.com/user-attachments/assets/94a25677-0e17-4bf3-94f6-1c43e6e6fd4d" />


<img width="778" height="630" alt="image" src="https://github.com/user-attachments/assets/203af1e0-a7e9-4a31-b576-fafa1d32f2b6" />



📈 Visualization

Clusters are visualized using petal length vs petal width

Each data point is colored based on its assigned cluster

🔹 Hierarchical Clustering (Agglomerative)
🔍 How Hierarchical Clustering Works

Builds a hierarchy of clusters

Uses a bottom-up (agglomerative) approach

Each data point starts as an individual cluster

Closest clusters are merged iteratively

The process is visualized using a dendrogram

✅ Why Hierarchical Clustering is Suitable for the Iris Dataset

The dataset is small (150 samples), making it computationally feasible

Features are continuous and numerical

Natural cluster separation can be clearly observed

Does not require specifying the number of clusters initially

🌳 Dendrogram Analysis

A dendrogram is plotted using Ward’s linkage method

The dendrogram helps identify the optimal number of clusters

Based on the dendrogram, 3 clusters are selected


<img width="986" height="652" alt="image" src="https://github.com/user-attachments/assets/48c0765e-fa9b-417b-9d62-1f04d0d775e6" />



📈 Visualization

Agglomerative clustering is applied using sklearn

Final clusters are visualized using scaled feature values

Minor overlaps are expected since clustering is performed in higher-dimensional space



<img width="967" height="665" alt="image" src="https://github.com/user-attachments/assets/30ae2d33-2267-441d-a5cd-233b50076e93" />



📌 Results & Observations

Both KMeans and Hierarchical clustering successfully group the Iris dataset

KMeans provides faster convergence with predefined clusters

Hierarchical clustering offers better interpretability through dendrograms

Minor overlaps in visualization are normal due to dimensionality reduction

🧾 Conclusion

This project demonstrates the application of two popular clustering algorithms on the Iris dataset.
Both methods effectively identify meaningful groups, with KMeans excelling in efficiency and Hierarchical clustering providing better insight into data structure.

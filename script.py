import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate synthetic data for clustering
cluster_data = {
    'Feature_1': np.random.uniform(0, 100, 100),
    'Feature_2': np.random.uniform(0, 100, 100)
}

df_cluster = pd.DataFrame(cluster_data)
kmeans = KMeans(n_clusters=3, random_state=42)
df_cluster['Cluster'] = kmeans.fit_predict(df_cluster)

# Scatter plot of clusters
plt.scatter(df_cluster['Feature_1'], df_cluster['Feature_2'], c=df_cluster['Cluster'], cmap='viridis')
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

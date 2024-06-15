from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import mglearn
import matplotlib.pyplot as plt
X, _ = make_blobs(n_samples=1000, centers=3,n_features=2)
print(X)
kmeans = KMeans(n_clusters=3, random_state=0,n_init="auto").fit(X)
print("data points belongs to clusters",kmeans.labels_)
print("cluster centroids are asfollows",kmeans.cluster_centers_)
pred, _ = make_blobs(n_samples=1, centers=1,n_features=2)
print("new data points",pred)
print("new clusters belong to ",kmeans.predict(pred))
# Plot original data points with cluster colors and
cluster centroids
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_,cmap='viridis', s=50, alpha=0.5, label='Data points')
plt.scatter(kmeans.cluster_centers_[:, 0],
kmeans.cluster_centers_[:, 1], marker='^', c='red',s=100, label='Cluster centroids')
# Plot new data point
plt.scatter(pred[:, 0], pred[:, 1], marker='x', c='black',s=100, label='New data point')
# Add cluster labels
for i, center in enumerate(kmeans.cluster_centers_):plt.text(center[0], center[1], f'Cluster {i}',fontsize=12, color='red', ha='center')
# Add legend and labels
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('KMeans Clustering')
plt.legend()
plt.show()
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.neighbors import NearestNeighbors

# Data
task_state1 = np.load('robot_env/task_states_push1_embeddings.npy') # (500, 2054)
task_state2 = np.load('robot_env/task_states_push4_embeddings.npy') 
anchors = np.load('robot_env/anchor_states_14_embeddings.npy')  

data_all = np.vstack((task_state1, task_state2, anchors))

num_samples_per_task_state = 500
num_anchors = anchors.shape[0] 
print(num_anchors)
labels = ['Task 1'] * num_samples_per_task_state + ['Task 2'] * num_samples_per_task_state + ['Anchor'] * num_anchors
colors = ['red'] * num_samples_per_task_state + ['blue'] * num_samples_per_task_state + ['green'] * num_anchors

pca = PCA(n_components=4)
data_reduced = pca.fit_transform(data_all)


task_state1_reduced = data_reduced[:num_samples_per_task_state, :]
task_state2_reduced = data_reduced[num_samples_per_task_state:2*num_samples_per_task_state, :]
anchors_reduced = data_reduced[-num_anchors:, :]

# Plotting
plt.figure(figsize=(12, 8))

# Plot each category with its own color and label
plt.scatter(task_state1_reduced[:, 0], task_state1_reduced[:, 1], c='red', label='Push 1', s=20, alpha=0.6)
plt.scatter(task_state2_reduced[:, 0], task_state2_reduced[:, 1], c='blue', label='Push 4', s=20, alpha=0.6)
plt.scatter(anchors_reduced[:, 0], anchors_reduced[:, 1], c='green', label='Anchor', s=10, alpha=0.6)

plt.title("Latent Space Visualization via PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()

tsne = TSNE(n_components=2, random_state=0)
scaler = StandardScaler()

# Fit the scaler to the data and then transform it
data_all_scaled = scaler.fit_transform(data_all)
data_tsne = tsne.fit_transform(data_all_scaled)  

task_state1_tsne = data_tsne[:500, :]
task_state2_tsne = data_tsne[500:1000, :]
anchors_tsne = data_tsne[-128:, :]

# Plotting
plt.figure(figsize=(10, 7))
plt.scatter(task_state1_tsne[:, 0], task_state1_tsne[:, 1], c='red', label='Push 1', s=20, alpha=0.7)
plt.scatter(task_state2_tsne[:, 0], task_state2_tsne[:, 1], c='blue', label='Push 4', s=20, alpha=0.7)
plt.scatter(anchors_tsne[:, 0], anchors_tsne[:, 1], c='green', label='Anchor', s=50, alpha=0.9)
plt.title("Latent Space Visualization via t-SNE")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.show()

data_all = np.vstack((task_state1, task_state2))
pca = PCA(n_components=4)  
data_all_scaled = scaler.fit_transform(data_all)
data_reduced = pca.fit_transform(data_all_scaled)
# On original data
n_clusters = 4  
clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors',n_neighbors=250, assign_labels="discretize", random_state=0)
labels = clustering.fit_predict(data_all_scaled)

# Centroids
centroids = np.zeros((n_clusters, data_all_scaled.shape[1]))  # Initialize an array to store centroids

for i in range(n_clusters):
    # Select all data points assigned to cluster i and calculate their mean
    centroids[i] = np.mean(data_all_scaled[labels == i], axis=0)
centroids_reduced = pca.transform(centroids)

nn = NearestNeighbors(n_neighbors=1, metric='manhattan')
nn.fit(data_all_scaled)

# Find the nearest neighbors for the centroids
distances, indices = nn.kneighbors(centroids)
breakpoint()
# 2054 for images, 6 for no images
closest_anchors = np.zeros((4, 2054))

for idx in range(len(centroids)):
    #breakpoint()
    closest_anchors[idx] = data_all[indices[idx]]
anchors_reduced = pca.transform(closest_anchors)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=labels, cmap='viridis', alpha=0.7, edgecolors='w')
plt.scatter(centroids_reduced[:, 0], centroids_reduced[:, 1], c='red', marker='X', s=100)  # Centroids in red
plt.scatter(anchors_reduced[:, 0], anchors_reduced[:, 1], c='black', marker='X', s=100)  # Centroids in red
plt.title('Spectral Clustering Results- Push1 and Push4')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')


# Add a legend
plt.legend(handles=scatter.legend_elements()[0], title="Clusters", labels=range(n_clusters))

plt.show()

pca = PCA(n_components=4)
data_reduced = pca.fit_transform(data_all_scaled)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot in 3D
scatter = ax.scatter(data_reduced[:, 0], data_reduced[:, 1], data_reduced[:, 2], c=labels, cmap='viridis', alpha=0.7, edgecolor='w')
ax.scatter(centroids_reduced[:, 0], centroids_reduced[:, 1], c='red', marker='X', s=100)  # Centroids in red
ax.scatter(anchors_reduced[:, 0], anchors_reduced[:, 1], c='black', marker='X', s=100)  # Centroids in red
ax.set_title('Spectral Clustering Results in 3D- Push1 and Push4')
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')

# Legend
ax.legend(handles=scatter.legend_elements()[0], title="Clusters", labels=range(n_clusters))

plt.show()

np.save('spectral_clustering_centroids', closest_anchors)

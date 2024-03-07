from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
task_state1 = np.load('robot_env/task_states_push1_embeddings.npy')
task_state2 = np.load('robot_env/task_states_push4_embeddings.npy')
# print(task_state[-20:-10])
task_state_mix = np.concatenate((task_state1, task_state2))
print(task_state_mix.shape)

kmeans = KMeans(n_clusters=2054, random_state=0).fit(task_state_mix)
#
center_anchors = kmeans.cluster_centers_ # Gets the center anchors
# print(center_anchors)
print(center_anchors.shape)

np.save('kmeans_centers.npy', center_anchors)

center_anchors = np.load('kmeans_centers.npy') # Loads them

# get the 128 task states which are closest to the 128 kmeans centers
closest, _ = pairwise_distances_argmin_min(center_anchors, task_state_mix) # Closest task states to centers- why?
breakpoint()
# 2054 for images, 6 for no images
closest_anchors = np.zeros((2054, 2054))
print(f"closest size: {closest.shape}")
print(f"Center anchors size {center_anchors.shape}")
print(f"task state mix size: {task_state_mix.shape}")

for idx in range(len(closest)):
    #breakpoint()
    closest_anchors[idx] = task_state_mix[closest[idx]]

# real task states that are closest to the 128 centroids
np.save('anchor_states_14_embeddings', closest_anchors)

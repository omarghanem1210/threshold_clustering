import numpy as np
from scipy.sparse.csgraph import laplacian
from sklearn.cluster import SpectralClustering

def eigengap_heuristic(adjacency_matrix):
    lap = laplacian(adjacency_matrix)
    eigen_values = sorted(np.unique(np.linalg.eig(lap)[0]))
    diff_eigen = np.diff(eigen_values, 1)
    max_value = np.max(diff_eigen)
    return np.where(diff_eigen==max_value)[0][0]


def threshold_clustering(data, q, num_clusters=None):
    n = len(data)
    data = data.T
    adjacency_matrix = np.zeros(shape=(n, n))
    Z = np.zeros(shape=(n, n))
    indices = list(range(n))

    for i in indices:
        current_node = data[i]
        other_nodes_indices = indices[0:i] + indices[i+1:]
        comp_func = lambda x: np.abs(np.dot(current_node, data[x])) # Comparison function for sorting
        nearest_neighbors_indices = sorted(other_nodes_indices, reverse=True, key=comp_func)[0:q]
        z = [np.exp(-2 *np.arccos(np.abs(np.dot(current_node, data[i])))) if i in nearest_neighbors_indices else 0 for i in indices]
        Z[i] = z

    adjacency_matrix = Z + Z.T
    if not num_clusters:
        num_clusters = eigengap_heuristic(adjacency_matrix)
    
    # Spectral Clustering
    sc = SpectralClustering(num_clusters, affinity='precomputed', n_init=100)
    sc.fit(adjacency_matrix)

    return sc.labels_



import numpy as np
from sklearn.neighbors import NearestNeighbors


def determine_optimal_epsilon(arr, min_samples=2):
    """
    Determines the optimal epsilon value for clustering using the k-distance graph method.

    Parameters:
    arr (np.ndarray): The input array of float values.
    min_samples (int): The number of nearest neighbors to consider.

    Returns:
    float: The optimal epsilon value.
    """
    # Reshape the array for NearestNeighbors
    arr = arr.reshape(-1, 1)

    # Compute the nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(arr)
    distances, indices = neighbors_fit.kneighbors(arr)

    # Sort the distances to the nearest neighbors
    distances = np.sort(distances[:, 1], axis=0)

    # Find the point of maximum curvature (elbow)
    # This is a simple heuristic; more sophisticated methods can be used
    diff = np.diff(distances)
    optimal_index = np.argmax(diff)
    optimal_epsilon = distances[optimal_index]

    return optimal_epsilon


def cluster_1d_array(arr, min_samples=2):
    """
    Clusters a 1D array of float values such that adjacent values are grouped together.

    Parameters:
    arr (list or np.ndarray): The input array of float values.
    min_samples (int): The number of nearest neighbors to consider for determining epsilon.

    Returns:
    tuple: A tuple containing a list of clusters and a dictionary mapping each index of the input array to the index of the identified cluster.
    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    # Determine the optimal epsilon value
    eps = determine_optimal_epsilon(arr, min_samples)

    # Sort the array and keep track of original indices
    sorted_indices = np.argsort(arr)
    sorted_arr = arr[sorted_indices]

    # Initialize clusters and index mapping
    clusters = []
    index_mapping = {}
    current_cluster = [sorted_arr[0]]
    current_cluster_index = 0

    # Iterate through the sorted array and form clusters
    for i in range(1, len(sorted_arr)):
        if sorted_arr[i] - sorted_arr[i - 1] <= eps:
            current_cluster.append(sorted_arr[i])
        else:
            clusters.append(current_cluster)
            for idx in sorted_indices[i - len(current_cluster) : i]:
                index_mapping[idx] = current_cluster_index
            current_cluster = [sorted_arr[i]]
            current_cluster_index += 1

    # Append the last cluster
    clusters.append(current_cluster)
    for idx in sorted_indices[len(sorted_arr) - len(current_cluster) :]:
        index_mapping[idx] = current_cluster_index

    return clusters, index_mapping

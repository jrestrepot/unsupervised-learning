import numpy as np
import pandas as pd

from unsupervised.clustering_algorithms.connected_components import (
    ConnectedComponentsCluster,
)
from unsupervised.clustering_algorithms.distance_cluster import DistanceCluster
from unsupervised.clustering_algorithms.knn import KNN
from unsupervised.distances import mahalanobis_distance
from unsupervised.utils import create_nd_grid, get_params, process_data, read_data

PARAMETERS_FILE = "parameters.json"

if __name__ == "__main__":
    # Set the parameters
    (
        DATA_PATH,
        DISTANCE,
        DISTANCE_KWARGS,
        DISTANCE_THRESHOLD,
        K,
        N_CLUSTERS,
    ) = get_params(PARAMETERS_FILE)

    # Read data
    data = read_data(DATA_PATH)
    # Drop the id and species columns
    data.drop(["Id", "Species"], axis=1, inplace=True)
    # Process the data
    data = process_data(data)

    # Set distance kwargs for the mahalanobis distance
    if DISTANCE == mahalanobis_distance:
        DISTANCE_KWARGS["cov"] = np.cov(data.T)

    # Compute the connected components clusters
    print("Connected Components")
    connected_components = ConnectedComponentsCluster(
        data, DISTANCE_THRESHOLD, DISTANCE, DISTANCE_KWARGS
    )
    clusters = connected_components.predict()
    connected_components.plot_clusters(
        f"{DISTANCE_THRESHOLD}_dist_{DISTANCE.__name__}_{DISTANCE_KWARGS.get('p', '')}"
    )

    # Compute the KNN clusters
    print("KNN")
    knn = KNN(data, K, DISTANCE, DISTANCE_KWARGS)
    knn.plot_clusters(f"{K}_{DISTANCE.__name__}_{DISTANCE_KWARGS.get('p', '')}")

    # Compute the distance clusters
    print("Distance Clusters")
    dist_cluster = DistanceCluster(data, N_CLUSTERS, DISTANCE, DISTANCE_KWARGS)

# import subprocess

# # Install the requirements
# subprocess.run(["pip", "install", "-r", "requirements.txt"])
# from pprint import pprint

import numpy as np
import pandas as pd
from clustering.connected_components import ConnectedComponentsCluster
from clustering.distance_cluster import DistanceCluster
from clustering.knn import KNN
from distances import cosine_distance, lp_distance, mahalanobis_distance
from utils import create_nd_grid, process_data, read_data

# The distances or norms to use for the clustering
DISTANCE = lp_distance
# The keyword arguments for the distance function (the p for the lp distance, leave
# empty for the others)
DISTANCE_KWARGS = {"p": 2}

# The distance threshold for the connected components clustering
DISTANCE_THRESHOLD = 0.1
# The number of nearest neighbors for the KNN clustering
K = 50
# The number of clusters for the distance clustering
N_CLUSTERS = 3

if __name__ == "__main__":
    # Read data
    data = read_data("unsupervised/data/Iris.csv")
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

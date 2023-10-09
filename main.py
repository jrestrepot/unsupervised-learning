from pprint import pprint

import numpy as np

from unsupervised.autoencoder import Autoencoder
from unsupervised.clustering_algorithms.connected_components import (
    ConnectedComponentsCluster,
)
from unsupervised.clustering_algorithms.distance_cluster import DistanceCluster
from unsupervised.clustering_algorithms.knn import KNN
from unsupervised.descriptive import pairplot
from unsupervised.distances import mahalanobis_distance
from unsupervised.utils import (
    create_nd_grid,
    describe_analyze_data,
    get_params,
    process_data,
    read_data,
)

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
    # Drop the id column
    data.drop(["Id"], axis=1, inplace=True)
    pairplot(data, "Species")

    # Drop the species column
    data.drop(["Species"], axis=1, inplace=True)
    # Process the data
    data = process_data(data)

    # Describe and analyze the data
    describe_analyze_data(data)
    data = data.to_numpy()

    # Compute the autoencoder low dimensional representation
    autoencoder = Autoencoder(data, 4, [2])
    autoencoder.fit()
    latent_space_low = autoencoder.encode()

    # Compute the autoencoder high dimensional representation
    autoencoder = Autoencoder(data, 4, [6])
    autoencoder.fit()
    latent_space_high = autoencoder.encode()

    # Set distance kwargs for the mahalanobis distance
    if DISTANCE == mahalanobis_distance:
        DISTANCE_KWARGS["cov"] = np.cov(data.T)

    # Compute the connected components clusters
    connected_components = ConnectedComponentsCluster(
        data, DISTANCE_THRESHOLD, DISTANCE, DISTANCE_KWARGS
    )
    connected_components.plot_clusters(
        f"{DISTANCE_THRESHOLD}_dist_{DISTANCE.__name__}_{DISTANCE_KWARGS.get('p', '')}"
    )
    print("")

    # Compute the KNN clusters
    knn = KNN(data, K, DISTANCE, DISTANCE_KWARGS)
    knn.plot_clusters(f"{K}_{DISTANCE.__name__}_{DISTANCE_KWARGS.get('p', '')}")
    print("")

    # Compute the distance clusters
    dist_cluster = DistanceCluster(data, N_CLUSTERS, DISTANCE, DISTANCE_KWARGS)
    print("This clustering algorithm doesn't have a plot method.")

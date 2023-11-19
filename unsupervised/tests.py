import concurrent.futures
from typing import Callable

import numpy as np
from tqdm import tqdm

from unsupervised.autoencoder import Autoencoder
from unsupervised.clustering_algorithms.connected_components import (
    ConnectedComponentsCluster,
)
from unsupervised.clustering_algorithms.distance_cluster import DistanceCluster
from unsupervised.clustering_algorithms.fuzzy_cmeans import FuzzyCMeans
from unsupervised.clustering_algorithms.kmeans import KMeans
from unsupervised.clustering_algorithms.mountain_clustering import MountainClustering
from unsupervised.clustering_algorithms.subtractive_clustering import (
    SubtractiveClustering,
)
from unsupervised.distances import mahalanobis_distance
from unsupervised.utils import cluster_validation, create_latex_table, pairwise_distance


def test_distance_clusters(
    data: np.ndarray,
    DISTANCES: list[Callable],
    DISTANCE_KWARGS: list[dict],
    N_CLUSTERS: list[int],
    name: str = "distance_clusters",
):
    """
    A function to test the distance clusters algorithm with various hyperparameters.

    Arguments:
    ----------
    data: np.ndarray
        The data to cluster.
    DISTANCES: list[Callable]
        The list of distance functions to use.
    DISTANCES_KWARGS: list[dict]
        The list of distance functions keyword arguments.
    N_CLUSTERS: list[int]
        The list of number of clusters to use.
    """

    results = []
    for DISTANCE, DISTANCE_KWARGS in zip(
        tqdm(DISTANCES, desc="Testing Distance Clustering"), DISTANCE_KWARGS
    ):
        # If the distance is mahalanobis, we need to pass the covariance matrix
        if DISTANCE == mahalanobis_distance:
            DISTANCE_KWARGS["cov"] = np.cov(data.T)

        for N_CLUSTER in range(N_CLUSTERS[0], N_CLUSTERS[1] + 1, N_CLUSTERS[2]):
            distance_cluster = DistanceCluster(
                data,
                N_CLUSTER,
                DISTANCE,
                DISTANCE_KWARGS,
            )
            obs_index = np.random.randint(0, data.shape[0])
            clusters = distance_cluster.predict(obs_index)
            # Get labels from clusters
            labels = np.zeros(data.shape[0])
            for i, index in enumerate(clusters):
                labels[index] = i
            distance_matrix = pairwise_distance(data, data, DISTANCE, DISTANCE_KWARGS)
            validation = cluster_validation(
                distance_matrix,
                data,
                labels,
                Distance=DISTANCE.__name__,
                distance_kwargs=DISTANCE_KWARGS.get("p", ""),
                N=N_CLUSTER,
            )
            results.append(validation)
    create_latex_table(results, name)


def test_connected_components(
    data: np.ndarray,
    DISTANCES: list[Callable],
    DISTANCE_KWARGS: list[dict],
    THRESHOLDS: list[float],
    name: str = "connected_components",
):
    """
    A function to test the connected components algorithm with various hyperparameters.

    Arguments:
    ----------
    data: np.ndarray
        The data to cluster.
    DISTANCES: list[Callable]
        The list of distance functions to use.
    DISTANCES_KWARGS: list[dict]
        The list of distance functions keyword arguments.
    THRESHOLDS: list[float]
        The list of thresholds to use.
    """

    results = []
    for DISTANCE, DISTANCE_KWARGS, THRESHOLDS in zip(
        tqdm(DISTANCES, desc="Testing Connected Components ..."),
        DISTANCE_KWARGS,
        THRESHOLDS,
    ):
        # If the distance is mahalanobis, we need to pass the covariance matrix
        if DISTANCE == mahalanobis_distance:
            DISTANCE_KWARGS["cov"] = np.cov(data.T)
        for THRESHOLD in THRESHOLDS:
            connected_components = ConnectedComponentsCluster(
                data,
                THRESHOLD,
                DISTANCE,
                DISTANCE_KWARGS,
            )
            clusters = connected_components.predict()
            # Get labels from clusters
            labels = np.zeros(data.shape[0])
            for i, index in enumerate(clusters):
                labels[index] = i
            if len(clusters) == 1:
                continue
            distance_matrix = pairwise_distance(data, data, DISTANCE, DISTANCE_KWARGS)
            validation = cluster_validation(
                distance_matrix,
                data,
                labels,
                Distance=DISTANCE.__name__,
                distance_kwargs=DISTANCE_KWARGS.get("p", ""),
                N=len(clusters),
                Distance_Threshold=THRESHOLD,
            )
            results.append(validation)
    create_latex_table(results, name)


def test_mountain_clustering_parallel(
    data: np.ndarray,
    NUM_PARTITION: int,
    DISTANCE: Callable,
    DISTANCE_KWARGS: dict,
    SIGMA: float,
    BETA: float,
    M_FUZZY_CMEANS: int,
):
    """
    A helper function to test the mountain clustering algorithm in parallel
    """

    print("Testing Mountain Clustering ...")
    mountain = MountainClustering(
        data, NUM_PARTITION, DISTANCE, DISTANCE_KWARGS, SIGMA, BETA
    )
    centers = mountain.predict()
    if len(centers) == 1:
        (f"WARNING: Only one center found, {SIGMA}, {BETA}")
        return
    kmeans = KMeans(
        data, len(centers), DISTANCE, DISTANCE_KWARGS, initial_centers=centers
    )
    kmeans_results = kmeans.predict()
    fuzzy_cmeans = FuzzyCMeans(
        data,
        len(centers),
        DISTANCE,
        DISTANCE_KWARGS,
        m=M_FUZZY_CMEANS,
        initial_centers=centers,
    )
    _, membership_matrix = fuzzy_cmeans.predict()
    # Get hard clustering labels
    labels_fuzzy = np.argmax(membership_matrix, axis=0)
    if len(np.unique(labels_fuzzy)) == 1:
        print(f"WARNING: Only one cluster found, {SIGMA}, {BETA}")
        return

    kmeans_names, kmeans_val = cluster_validation(
        pairwise_distance(data, data, DISTANCE, DISTANCE_KWARGS),
        data,
        kmeans_results,
        Distance=DISTANCE.__name__,
        distance_kwargs=DISTANCE_KWARGS.get("p", ""),
        N=len(centers),
    )
    mountain_val = (
        [
            "Sigma",
            "Beta",
            "Num centers",
        ],
        [
            SIGMA,
            BETA,
            len(centers),
        ],
    )
    mountain_kmeans_names = mountain_val[0] + kmeans_names
    mountain_kmeans_val = mountain_val[1] + kmeans_val
    mountain_kmeans = (mountain_kmeans_names, mountain_kmeans_val)

    fuzzy_cmeans_names, fuzzy_cmeans_val = cluster_validation(
        pairwise_distance(data, data, DISTANCE, DISTANCE_KWARGS),
        data,
        labels_fuzzy,
        Distance=DISTANCE.__name__,
        distance_kwargs=DISTANCE_KWARGS.get("p", ""),
        N=len(centers),
        M=M_FUZZY_CMEANS,
    )

    mountain_fuzzy_names = mountain_val[0] + fuzzy_cmeans_names
    mountain_fuzzy_val = mountain_val[1] + fuzzy_cmeans_val
    mountain_fuzzy = (mountain_fuzzy_names, mountain_fuzzy_val)

    return mountain_kmeans, mountain_fuzzy


def test_mountain_clustering(
    data: np.ndarray,
    NUM_PARTITIONS: list[int],
    DISTANCES: list[Callable],
    DISTANCES_KWARGS: list[dict],
    SIGMAS: list[float],
    BETAS: list[float],
    M_FUZZY_CMEANS: list[int],
    name: str = "mountain_clustering",
):
    """
    A function to test the mountain clustering algorithm with various hyperparameters.

    Arguments:
    ----------
    data: np.ndarray
        The data to cluster.
    NUM_PARTITIONS: list[int]
        The list of number of partitions to use.
    SIGMAS: list[float]
        The list of sigmas to use.
    BETAS: list[float]
        The list of betas to use.
    """

    parameter_combinations = []
    for DISTANCE, DISTANCE_KWARGS in zip(DISTANCES, DISTANCES_KWARGS):
        # If the distance is mahalanobis, we need to pass the covariance matrix
        if DISTANCE == mahalanobis_distance:
            DISTANCE_KWARGS["cov"] = np.cov(data.T)
        for NUM_PARTITION in NUM_PARTITIONS:
            for SIGMA in SIGMAS:
                for BETA in BETAS:
                    for M in range(
                        M_FUZZY_CMEANS[0], M_FUZZY_CMEANS[1], M_FUZZY_CMEANS[2]
                    ):
                        parameter_combinations.append(
                            (
                                data,
                                NUM_PARTITION,
                                DISTANCE,
                                DISTANCE_KWARGS,
                                SIGMA,
                                BETA,
                                M,
                            )
                        )
    # Use ThreadPoolExecutor to parallelize the processing
    print("Testing Mountain Clustering in parallel...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map the function to the combinations using the executor
        results = list(
            executor.map(
                lambda args: test_mountain_clustering_parallel(*args),
                parameter_combinations,
            )
        )

    results = [result for result in results if result is not None]
    kmeans = [result[0] for result in results]
    fuzzy = [result[1] for result in results]
    create_latex_table(kmeans, name + "_kmeans")
    create_latex_table(fuzzy, name + "_fuzzy")


def test_subtractive_clustering_parallel(
    data: np.ndarray,
    DISTANCE: Callable,
    DISTANCE_KWARGS: dict,
    RA: float,
    RB: float,
    M_FUZZY_CMEANS: int,
):
    """
    A helper function to test the subtractive clustering algorithm in parallel
    """

    print("Testing Subtractive Clustering ...")
    subtractive = SubtractiveClustering(data, DISTANCE, DISTANCE_KWARGS, RA, RB, 100)
    centers = subtractive.predict()
    kmeans = KMeans(
        data, len(centers), DISTANCE, DISTANCE_KWARGS, initial_centers=centers
    )
    kmeans_results = kmeans.predict()
    fuzzy_cmeans = FuzzyCMeans(
        data,
        len(centers),
        DISTANCE,
        DISTANCE_KWARGS,
        m=M_FUZZY_CMEANS,
        initial_centers=centers,
    )
    _, membership_matrix = fuzzy_cmeans.predict()
    # Get hard clustering labels
    labels_fuzzy = np.argmax(membership_matrix, axis=0)
    if len(np.unique(labels_fuzzy)) == 1:
        print(f"WARNING: Only one cluster found, {RA}, {RB}")
        return
    kmeans_names, kmeans_val = cluster_validation(
        pairwise_distance(data, data, DISTANCE, DISTANCE_KWARGS),
        data,
        kmeans_results,
        Distance=DISTANCE.__name__,
        distance_kwargs=DISTANCE_KWARGS.get("p", ""),
        N=len(centers),
    )
    subtractive_val = (
        ["Ra", "Rb", "Num centers"],
        [RA, RB, len(centers)],
    )
    subtractive_kmeans_names = subtractive_val[0] + kmeans_names
    subtractive_kmeans_val = subtractive_val[1] + kmeans_val
    subtractive_kmeans = (subtractive_kmeans_names, subtractive_kmeans_val)

    fuzzy_cmeans_names, fuzzy_cmeans_val = cluster_validation(
        pairwise_distance(data, data, DISTANCE, DISTANCE_KWARGS),
        data,
        labels_fuzzy,
        Distance=DISTANCE.__name__,
        distance_kwargs=DISTANCE_KWARGS.get("p", ""),
        N=len(centers),
        M=M_FUZZY_CMEANS,
    )
    subtractive_fuzzy_names = subtractive_val[0] + fuzzy_cmeans_names
    subtractive_fuzzy_val = subtractive_val[1] + fuzzy_cmeans_val
    subtractive_fuzzy = (subtractive_fuzzy_names, subtractive_fuzzy_val)
    return subtractive_kmeans, subtractive_fuzzy


def test_subtractive_clustering(
    data: np.ndarray,
    DISTANCES: list[Callable],
    DISTANCES_KWARGS: list[dict],
    RAS: list[float],
    RBS: list[float],
    M_FUZZY_CMEANS: list[int],
    name: str = "subtractive_clustering",
):
    """
    A function to test the subtractive clustering algorithm with various hyperparameters.

    Arguments:
    ----------
    data: np.ndarray
        The data to cluster.
    DISTANCES: list[Callable]
        The list of distance functions to use.
    DISTANCES_KWARGS: list[dict]
        The list of distance functions keyword arguments.
    RAS: list[float]
        The list of ra values to use.
    RBS: list[float]
        The list of rb values to use.
    """

    parameter_combinations = []
    for DISTANCE, DISTANCE_KWARGS in zip(DISTANCES, DISTANCES_KWARGS):
        # If the distance is mahalanobis, we need to pass the covariance matrix
        if DISTANCE == mahalanobis_distance:
            DISTANCE_KWARGS["cov"] = np.cov(data.T)
        for RA, RB in zip(RAS, RBS):
            for M in range(M_FUZZY_CMEANS[0], M_FUZZY_CMEANS[1], M_FUZZY_CMEANS[2]):
                parameter_combinations.append(
                    (data, DISTANCE, DISTANCE_KWARGS, RA, RB, M)
                )
    # Use ThreadPoolExecutor to parallelize the processing
    print("Testing Subtractive Clustering in parallel...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map the function to the combinations using the executor
        results = list(
            executor.map(
                lambda args: test_subtractive_clustering_parallel(*args),
                parameter_combinations,
            )
        )
    results = [result for result in results if result is not None]
    kmeans = [result[0] for result in results]
    fuzzy = [result[1] for result in results]
    create_latex_table(kmeans, name + "_kmeans")
    create_latex_table(fuzzy, name + "_fuzzy")


def test_high_dimensions(
    data: np.ndarray,
    DISTANCES: list[Callable],
    DISTANCE_KWARGS: list[dict],
    DISTANCE_THRESHOLDS: list[float],
    M_FUZZY_CMEANS: list[int],
    N_CLUSTERS_DISTANCES: list[int],
    NUM_PARTITIONS: list[int],
    SIGMA: list[float],
    BETA: list[float],
    RA: list[float],
    RB: list[float],
):
    """
    A function to test the clustering algorithms on the high dimensional data.
    """

    print("====================================================")
    print("==============TESTING ON HIGH DIMENSIONS============")
    autoencoder = Autoencoder(data, data.shape[1], [6])
    autoencoder.fit()
    # Get the low dimensional data
    data = autoencoder.encode()
    test_distance_clusters(
        data,
        DISTANCES,
        DISTANCE_KWARGS,
        N_CLUSTERS_DISTANCES,
        name="distance_clusters_6",
    )
    test_connected_components(
        data,
        DISTANCES,
        DISTANCE_KWARGS,
        DISTANCE_THRESHOLDS,
        name="connected_components_6",
    )
    # Test algorithms to find centers with various hyperparameters
    test_mountain_clustering(
        data,
        NUM_PARTITIONS,
        DISTANCES,
        DISTANCE_KWARGS,
        SIGMA,
        BETA,
        M_FUZZY_CMEANS,
        name="mountain_clustering_6",
    )
    test_subtractive_clustering(
        data,
        DISTANCES,
        DISTANCE_KWARGS,
        RA,
        RB,
        M_FUZZY_CMEANS,
        name="subtractive_clustering_6",
    )


def test_low_dimensions(
    data: np.ndarray,
    DISTANCES: list[Callable],
    DISTANCE_KWARGS: list[dict],
    DISTANCE_THRESHOLDS: list[float],
    M_FUZZY_CMEANS: list[int],
    N_CLUSTERS_DISTANCES: list[int],
    NUM_PARTITIONS: list[int],
    SIGMA: list[float],
    BETA: list[float],
    RA: list[float],
    RB: list[float],
):
    """
    A function to test the clustering algorithms on the low dimensional data.
    """

    print("====================================================")
    print("==============TESTING ON LOW DIMENSIONS==============")
    # Create an autoencoder
    autoencoder = Autoencoder(data, data.shape[1], [2])
    autoencoder.fit()
    # Get the low dimensional data
    data = autoencoder.encode()

    test_distance_clusters(
        data,
        DISTANCES,
        DISTANCE_KWARGS,
        N_CLUSTERS_DISTANCES,
        name="distance_clusters_2",
    )
    test_connected_components(
        data,
        DISTANCES,
        DISTANCE_KWARGS,
        DISTANCE_THRESHOLDS,
        name="connected_components_2",
    )
    # Test algorithms to find centers with various hyperparameters
    test_mountain_clustering(
        data,
        NUM_PARTITIONS,
        DISTANCES,
        DISTANCE_KWARGS,
        SIGMA,
        BETA,
        M_FUZZY_CMEANS,
        name="mountain_clustering_2",
    )
    test_subtractive_clustering(
        data,
        DISTANCES,
        DISTANCE_KWARGS,
        RA,
        RB,
        M_FUZZY_CMEANS,
        name="subtractive_clustering_2",
    )


def test_original_dimensions(
    data: np.ndarray,
    DISTANCES: list[Callable],
    DISTANCE_KWARGS: list[dict],
    DISTANCE_THRESHOLDS: list[float],
    M_FUZZY_CMEANS: list[int],
    N_CLUSTERS_DISTANCES: list[int],
    NUM_PARTITIONS: list[int],
    SIGMA: list[float],
    BETA: list[float],
    RA: list[float],
    RB: list[float],
):
    """
    A function to test the clustering algorithms on the original data.
    """

    print("====================================================")
    print("==============TESTING ON ORIGINAL DATA==============")
    test_distance_clusters(
        data,
        DISTANCES,
        DISTANCE_KWARGS,
        N_CLUSTERS_DISTANCES,
        name="distance_clusters_original",
    )
    test_connected_components(
        data,
        DISTANCES,
        DISTANCE_KWARGS,
        DISTANCE_THRESHOLDS,
        name="connected_components_original",
    )
    test_subtractive_clustering(
        data,
        DISTANCES,
        DISTANCE_KWARGS,
        RA,
        RB,
        M_FUZZY_CMEANS,
        name="subtractive_clustering_original",
    )
    test_mountain_clustering(
        data,
        NUM_PARTITIONS,
        DISTANCES,
        DISTANCE_KWARGS,
        SIGMA,
        BETA,
        M_FUZZY_CMEANS,
        name="mountain_clustering_original",
    )

import json
from typing import Callable

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from unsupervised.descriptive import (
    biplot,
    descriptive_statistics,
    pairplot,
    plot_boxplot,
    plot_histogram,
    umap,
)
from unsupervised.distances import match_distance


def read_data(file_path: str) -> pd.DataFrame:
    """Reads the data from a file. It considers numerous file types: csv, txt,
    xlsx, xls, and json.

    Parameters
    ----------
    file_path: str
        The file path.

    Returns
    -------
    pd.DataFrame
        The data.
    """

    if file_path.endswith(".csv") or file_path.endswith(".txt"):
        data = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        data = pd.read_excel(file_path)
    elif file_path.endswith(".json"):
        data = pd.read_json(file_path)
    else:
        raise ValueError(f"File type not supported: {file_path}")
    return data


def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """It processes the DataFrame.

    Parameters
    ----------
    data: pd.DataFrame
        The data.

    Returns
    -------
    np.ndarray
        The processed data.
    """

    # One hot encode
    data = pd.get_dummies(data)
    # Drop the nans
    data = data.dropna()
    # Normalise with min max
    data = (data - data.min()) / (data.max() - data.min())
    return data


def pairwise_distance(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    distance: Callable,
    distance_kwargs: dict = None,
) -> np.ndarray:
    """
     Compute the pairwise distance between two matrices.

    Parameters
    ----------
    matrix_a: np.ndarray
        The first matrix.
    matrix_b: np.ndarray
        The second matrix.
    """

    assert matrix_a.shape[1] == matrix_b.shape[1]

    if distance_kwargs is None:
        distance_kwargs = {}
    if distance.__name__ == "mahalanobis_distance":
        distance_kwargs["cov"] = np.cov(matrix_a.T)

    # Compute the pairwise distance between two matrices
    distance_matrix = np.zeros((matrix_a.shape[0], matrix_b.shape[0]))
    for i in range(matrix_a.shape[0]):
        for j in range(matrix_b.shape[0]):
            distance_matrix[i, j] = distance(
                vector_a=matrix_a[i], vector_b=matrix_b[j], **distance_kwargs
            )
    return distance_matrix


def plot_distance_matrix(distances: np.ndarray, name: str) -> None:
    """
    It plots the distance matrix.

    Parameters
    ----------
    distances: np.ndarray
        The distance matrix.
    """

    fig = px.imshow(distances)
    # Save to png
    fig.write_html(f"results/distance_matrix_{name }.html")


def create_nd_grid_list(indices):
    return np.array(indices)


def create_nd_grid(num_partitions: int, dimension: int):
    """
    Create an n-dimensional grid with the specified shape.

    Args:
    dimension: int
        The dimension of the grid.
    num_partitions: int
        The number of partitions in each dimension.

    Returns:
    list: An n-dimensional grid represented as a list of tuples.
    """

    interval = np.linspace(0, 1, num_partitions)
    tiles = np.tile(interval, dimension).reshape(-1, num_partitions)

    return np.array(np.meshgrid(*tiles)).T.reshape(-1, dimension)


def get_params(file_path: str) -> tuple:
    """Get the parameters from the json file. It returns the default parameters if
    they don't exist in the file.

    The default parameters are:
    - DATA_PATH: "data/iris.csv"
    - DISTANCE: lp_distance
    - DISTANCE_KWARGS: {}
    - DISTANCE_THRESHOLD: 0.1
    - K: 10
    - N_CLUSTERS: 3

    Parameters
    ----------
    file_path: str
        The file path.

    Returns
    -------
    tuple
        The parameters.
    """

    with open(file_path) as f:
        params = json.load(f)
    if "distance" not in params:
        raise ValueError("The distance function is not specified.")
    if not isinstance(params["distance"], list):
        raise ValueError("The distance parameter must be a list.")
    distances = []
    for distance in params["distance"]:
        distances.append(match_distance(distance))
    return (
        params.get("data_path", "data/iris.csv"),
        distances,
        params.get("distance_kwargs", [{}]),
        params["connected_components"].get("distance_threshold", [0.1]),
        params["knn"].get("k", [10]),
        params["kmeans"].get("n_clusters", [3]),
        params["fuzzy_cmeans"].get("n_clusters", [3]),
        params["fuzzy_cmeans"].get("m", [2]),
        params["distance_clusters"].get("n_clusters", [3]),
        params["mountain_clustering"].get("num_partitions", [5]),
        params["mountain_clustering"].get("sigma", [0.5]),
        params["mountain_clustering"].get("beta", [0.8]),
        params["subtractive_clustering"].get("ra", [0.5]),
    )


def describe_analyze_data(data: pd.DataFrame | np.ndarray) -> None:
    """A function to describe and analyze the data.

    Arguments:
    ---------
        data (pd.DataFrame | np.ndarray):
            The data to describe and analyze.
    """

    print("Descriptive Statistics:")
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)

    # Compute descriptive statistics
    descriptive_stats = descriptive_statistics(data)
    print(descriptive_stats)

    # Plot histograms
    for column in data.columns:
        plot_histogram(data, column)

    # Plot boxplots
    for column in data.columns:
        plot_boxplot(data, column)

    # Plot PCA
    biplot(data)


def cluster_validation(
    distance_matrix: np.ndarray, data: np.ndarray, cluster_lables: np.ndarray, **kwargs
) -> tuple:
    """A function to validate the clustering results.

    Arguments:
    ---------
        distance_matrix (np.ndarray):
            The pairwise distance matrix.
        data (np.ndarray):
            The data.
        cluster_labels (np.ndarray):
            The cluster labels.
        kwargs:
            The clustering parameters.

    Returns:
    --------
        tuple:
            The clustering parameters and the validation results.
    """

    if any(np.isnan(distance_matrix.flatten())):
        print(distance_matrix)
        raise ValueError("The distance matrix contains NaN values.")
    # Compute the silhouette score
    silhouette_score_value = silhouette_score(distance_matrix, cluster_lables)
    # Compute the Davies-Bouldin score
    davies_bouldin_score_value = davies_bouldin_score(data, cluster_lables)
    # Compute the Calinski-Harabasz score
    calinski_harabasz_score_value = calinski_harabasz_score(data, cluster_lables)

    parameters_name = list(kwargs.keys())
    parameters_name.extend(
        [
            "Silhouette",
            "DB",
            "CH",
        ]
    )
    parameters = list(kwargs.values())
    parameters.extend(
        [
            round(silhouette_score_value, 3),
            round(davies_bouldin_score_value, 3),
            round(calinski_harabasz_score_value, 3),
        ]
    )
    return (parameters_name, parameters)


def create_latex_table(cluster_validation_results: list[tuple], name: str) -> None:
    """A function to create a latex table from the cluster validation results.

    Arguments:
    ---------
        cluster_validation_results (list[tuple]):
            The cluster validation results.
        name (str):
            The name of the table.
    """

    results = []
    for columns, values in cluster_validation_results:
        assert len(columns) == len(values)
        results.append(values)

    table = pd.DataFrame(
        results,
        columns=columns,
    )

    table = table.to_latex(index=False)
    # Save to txt file
    with open(f"tables/{name}.txt", "w") as f:
        f.write(table)


def analyze_and_transform_data(data: pd.DataFrame, target_column: str) -> np.ndarray:
    """A function that saves the plots and descrptive analysis into files
    and then it transforms the data

    Parameters:
    -----------
    data: pd.DataFrame
        The original untransformed data
    target_column: str
        The column we supervise

    Returns:
    --------
    np.ndarray
        The transformed data
    """

    "Compute descriptive statistics"
    pairplot(data, target_column)
    umap_embedding = umap(data, target_column)
    # Drop the species column
    data.drop([target_column], axis=1, inplace=True)
    # Process the data
    data = process_data(data)
    umap_embedding = process_data(umap_embedding)
    # Describe and analyze the data
    describe_analyze_data(data)
    return data.to_numpy(), umap_embedding.to_numpy()

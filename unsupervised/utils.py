import itertools
import json
from multiprocessing import Pool, cpu_count
from typing import Callable

import numpy as np
import pandas as pd
import plotly.express as px

from unsupervised.descriptive import (
    biplot,
    descriptive_statistics,
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
    # Save to html
    fig.write_html(f"results/distance_matrix_{name}.html")


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
    return (
        params.get("data_path", "data/iris.csv"),
        match_distance(params.get("distance", "lp_distance")),
        params.get("distance_kwargs", {}),
        params["connected_components"].get("distance_threshold", 0.1),
        params["knn"].get("k", 10),
        params["kmeans"].get("n_clusters", 3),
        params["fuzzy_cmeans"].get("n_clusters", 3),
        params["distance_clusters"].get("n_clusters", 3),
        params["mountain_clustering"].get("num_partitions", 5),
        params["mountain_clustering"].get("sigma", 0.5),
        params["mountain_clustering"].get("beta", 0.8),
        params["subtractive_clustering"].get("ra", 0.5),
        params["subtractive_clustering"].get("rb", 1),
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

    # Plot UMAP
    umap(data)

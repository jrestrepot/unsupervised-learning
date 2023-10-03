"""A module for computing different distances."""

import numpy as np


def lp_distance(vector_a: np.ndarray, vector_b: np.ndarray, p: int = 2) -> float:
    """It computes the Lp distance between two points.

    Parameters
    ----------
    p: int
        The p value.
    vector_a: np.ndarray.
        The first point.
    vector_b: np.ndarray
        The second point.

    Returns
    -------
    distance: float
        The distance between the two points.
    """

    assert vector_a.shape == vector_b.shape
    # Compute the Lp distance between two points
    distance = np.sum(np.abs(vector_a - vector_b) ** p) ** (1 / p)
    return distance


def mahalanobis_distance(
    vector_a: np.ndarray, vector_b: np.ndarray, cov: np.ndarray
) -> float:
    """It computes the Mahalanobis distance between two points.

    Parameters
    ----------
    vector_a: np.ndarray
        The first point.
    vector_b: np.ndarray
        The second point.
    cov: np.ndarray
        The covariance matrix.

    Returns
    -------
    distance: float
        The distance between the two points.
    """
    assert vector_a.shape == vector_b.shape

    if not vector_a.shape[0] == cov.shape[0]:
        vector_a = vector_a.T
        vector_b = vector_b.T

    # Compute the inverse of the covariance matrix
    inv_cov = np.linalg.inv(cov)
    # Compute the Mahalanobis distance between two points
    distance = np.sqrt(
        np.dot(np.dot((vector_a - vector_b).T, inv_cov), (vector_a - vector_b))
    )
    return distance


def cosine_distance(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """It computes the cosine distance between two points.

    Parameters
    ----------
    vector_a: np.ndarray
        The first point.
    vector_b: np.ndarray
        The second point.

    Returns
    -------
    distance: float
        The distance between the two points.
    """

    assert vector_a.shape == vector_b.shape

    # Calculate the dot product of the two vectors
    dot_product = np.dot(vector_a, vector_b)

    # Calculate the Euclidean norms (magnitudes) of the vectors
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    # Calculate the cosine similarity
    cosine_similarity = dot_product / (norm_a * norm_b)

    # Calculate the cosine distance bvector_b subtracting from 1
    cosine_distance = 1 - cosine_similarity

    return cosine_distance

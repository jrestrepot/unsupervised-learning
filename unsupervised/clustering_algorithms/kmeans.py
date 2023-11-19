"""A module for clustering using kmeans algorithm."""

from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from unsupervised.distances import lp_distance
from unsupervised.utils import pairwise_distance


class KMeans:
    def __init__(
        self,
        data: pd.DataFrame | np.ndarray,
        n_clusters: int,
        distance: Callable = lp_distance,
        distance_kwargs: dict | None = None,
        initial_centers: np.ndarray | None = None,
        max_iterations: int = 100,
        tolerance: float = 1e-5,
    ):
        """Initialise the model.

        Arguments:
        ---------
            data (pd.DataFrame):
                The data to cluster.
            n_clusters (int):
                The number of clusters to use.
            distance (Callable, optional):
                The distance function to use. Defaults to lp_distance.
            distance_kwargs (dict, optional):
                The distance function keyword arguments. Defaults to None.
            initial_centers (np.ndarray, optional):
                The initial centers. Defaults to None.
            max_iterations (int, optional):
                The maximum number of iterations. Defaults to 100.
            tolerance (float, optional):
                The tolerance for convergence. Defaults to 1e-5.
        """

        self.data = np.array(data)
        self.distance = distance
        self.distance_kwargs = distance_kwargs
        self.n_clusters = n_clusters
        self.centers = initial_centers
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def get_membership_matrix(self) -> int:
        """Get the membership matrix (a matriz of 0s and 1s, 1 means a data point
        belongs to a center).

        Returns:
        -------
            np.ndarray:
                The membership matrix
        """

        membership_matrix = np.zeros((self.n_clusters, self.data.shape[0]))
        distance_matrix = pairwise_distance(
            self.centers, self.data, self.distance, self.distance_kwargs
        )
        # Getting the min distance is another way of viewing the function uij
        minimum_distances_index = np.argmin(distance_matrix, axis=0)
        points_index = np.arange(self.data.shape[0])
        membership_matrix[minimum_distances_index, points_index] = 1
        return membership_matrix, distance_matrix

    def loss_function(self, membership_matrix, distance_matrix):
        """Compute the loss function given a membership matrix and a distance matrix.

        Arguments:
        ---------
            membership_matrix (np.ndarray):
                The membership matrix.
            distance_matrix (np.ndarray):
                The distance matrix.

        Returns:
        -------
            float:
                The loss function
        """

        return np.sum(membership_matrix * distance_matrix**2)

    def update_centers(self, membership_matrix):
        """Update the centers given a membership matrix.

        Arguments:
        ---------
            membership_matrix (np.ndarray):
                The membership matrix.
        Returns:
        -------
            np.ndarray:
                The updated centers.
        """

        for i in range(len(self.centers)):
            self.centers[i] = (
                1
                / np.sum(membership_matrix[i])
                * np.sum(membership_matrix[i] * self.data.T, axis=1)
            )

    def predict(self) -> np.ndarray:
        """Predict the cluster of the given observation.

        Arguments:
        ---------
            obs_index (int):
                The index of the observation to predict.

        Returns:
        -------
            np.ndarray:
                The predicted clusters.
        """

        previous_loss = np.inf
        # Initiliaze centers randomly if not given
        if self.centers is None:
            self.centers = self.data[
                np.random.choice(self.data.shape[0], self.n_clusters, replace=False)
            ]
        else:
            one_iter = True
        for _ in range(self.max_iterations):
            membership_matrix, distance_matrix = self.get_membership_matrix()
            if one_iter:
                break
            self.update_centers(membership_matrix)
            loss = self.loss_function(membership_matrix, distance_matrix)
            if abs(previous_loss - loss) < self.tolerance:
                break
            previous_loss = loss
        labels = np.argmax(membership_matrix, axis=0)
        return labels

    def plot_clusters(self, example_name: str) -> None:
        """Plot the clusters. It only plots the first three dimensions.

        Arguments:
        ---------
            example_name (str):
                The name of the example.
        """

        example_name = example_name + f"{self.data.shape[1]}_dim"
        fig = go.Figure()
        labels = self.predict()
        clusters = [np.where(labels == i)[0] for i in range(self.n_clusters)]

        for cluster in clusters:
            # Plot the clusters
            fig.add_trace(
                go.Scatter3d(
                    x=self.data[cluster, 0],
                    y=self.data[cluster, 1],
                    z=self.data[cluster, 2],
                    mode="markers",
                    marker=dict(
                        size=3,
                        opacity=0.5,
                    ),
                )
            )
        # Save to png
        fig.write_html(f"results/kmeans_clusters_{example_name }.html")

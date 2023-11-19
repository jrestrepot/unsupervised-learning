"""A module for clustering using kmeans algorithm."""

from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from unsupervised.distances import lp_distance
from unsupervised.utils import pairwise_distance


class FuzzyCMeans:
    def __init__(
        self,
        data: pd.DataFrame | np.ndarray,
        n_clusters: int,
        distance: Callable = lp_distance,
        distance_kwargs: dict | None = None,
        initial_centers: np.ndarray | None = None,
        m: int = 2,
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
            m (int, optional):
                The fuzziness parameter. Defaults to 2.
            max_iterations (int, optional):
                The maximum number of iterations. Defaults to 100.
            tolerance (float, optional):
                The tolerance for convergence. Defaults to 1e-5.
        """

        assert m > 1, "m must be greater than 1"

        self.data = np.array(data)
        self.distance = distance
        self.distance_kwargs = distance_kwargs
        self.n_clusters = n_clusters
        self.m = m
        self.membership_matrix = None
        self.initial_centers = initial_centers
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def update_membership_matrix(self, distance_matrix: np.ndarray) -> int:
        """Update the membership matrix (a matriz of 0s and 1s, 1 means a data point
        belongs to a center).

        Arguments:
        ---------
            centers (np.ndarray):
                The centers to use.
            distance_matrix (np.ndarray):
                The distance matrix between the cesnters and the data.
        """

        # Check for a possible division by 0, this happens when a data point is
        # at the same position as a center
        rows, cols = np.where(distance_matrix == 0)
        if len(rows) > 0:
            distance_matrix[rows, cols] = 1e-10
        for cluster in range(self.n_clusters):
            self.membership_matrix[cluster] = 1 / np.sum(
                (distance_matrix[cluster] / (distance_matrix)) ** (2 / (self.m - 1)),
                axis=0,
            )

    def loss_function(self, centers: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute the loss function given a membership matrix and a distance matrix.

        Arguments:
        ---------
         centers (np.ndarray):
                The centers to use.

        Returns:
        -------
            float:
                The loss function
        """

        # Compute the distances between centers and data points
        distance_matrix = pairwise_distance(
            centers, self.data, self.distance, self.distance_kwargs
        )

        loss = np.sum(np.sum(self.membership_matrix**self.m * distance_matrix**2))
        return loss, distance_matrix

    def get_centers(self):
        """Update the centers.

        Returns:
        -------
            np.ndarray:
                The updated centers.
        """

        centers = np.zeros((self.n_clusters, self.data.shape[1]))
        for i in range(len(centers)):
            centers[i] = np.sum(
                self.membership_matrix[i] ** self.m * self.data.T, axis=1
            ) / np.sum(self.membership_matrix[i] ** self.m)

        return centers

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

        self.membership_matrix = np.random.rand(self.n_clusters, self.data.shape[0])
        for i in range(self.max_iterations):
            centers = self.get_centers()
            if i == 0 and self.initial_centers is not None:
                centers = self.initial_centers
            loss, distance_matrix = self.loss_function(centers)
            if abs(previous_loss - loss) < self.tolerance:
                break
            previous_loss = loss
            self.update_membership_matrix(distance_matrix)
            if self.initial_centers is not None:
                break
        return centers, self.membership_matrix

    def plot_clusters(self, example_name: str) -> None:
        """Plot the clusters. It only plots the first three dimensions.

        Arguments:
        ---------
            example_name (str):
                The name of the example.
        """

        example_name = example_name + f"{self.data.shape[1]}_dim"
        fig = go.Figure()
        centers, membership_matrix = self.predict()

        # Plot the centers
        fig.add_trace(
            go.Scatter3d(
                x=centers[:, 0],
                y=centers[:, 1],
                z=centers[:, 2],
                mode="markers",
                marker=dict(
                    size=3,
                    opacity=1,
                ),
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=self.data[:, 0],
                y=self.data[:, 1],
                z=self.data[:, 2],
                mode="markers",
                marker=dict(
                    size=3,
                    opacity=0.5,
                ),
            )
        )
        # Save to png
        fig.write_html(f"results/fuzzy_cmeans_centers{example_name }.html")

        # Plot the membership matrix
        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=membership_matrix.T,
                colorscale="Viridis",
            )
        )
        # Save to png
        fig.write_html(f"results/fuzzy_cmeans_matrix_{example_name }.html")

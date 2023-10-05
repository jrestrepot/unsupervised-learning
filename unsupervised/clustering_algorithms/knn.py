"""A module for k-nearest neighbors clustering."""

from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from unsupervised.distances import lp_distance
from unsupervised.utils import pairwise_distance


class KNN:
    def __init__(
        self,
        data: pd.DataFrame | np.ndarray,
        n_nearest: int,
        distance: Callable = lp_distance,
        distance_kwargs: dict | None = None,
    ):
        """Initialise the KNN model.

        Arguments:
        ---------
            data (pd.DataFrame):
                The data to cluster.
            n_nearest (int):
                The number of nearest neighbors to use.
            distance (Callable, optional):
                The distance function to use. Defaults to lp_distance.
            distance_kwargs (dict, optional):
                The distance function keyword arguments. Defaults to None.
        """

        self.data = np.array(data)
        self.n_nearest = n_nearest
        self.distance_matrix = pairwise_distance(
            self.data, self.data, distance, distance_kwargs
        )

    def predict(self, obs_index: int) -> np.ndarray:
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

        # Compute the distance between the data and the input
        distances = self.distance_matrix[obs_index]
        # Get the indices of the nearest neighbors
        nearest_indices = np.argsort(distances, axis=0)[: self.n_nearest]
        # Get the data of the nearest neighbors
        nearest_data = self.data[nearest_indices]

        return nearest_indices, nearest_data

    def plot_clusters(self, example_name: str) -> None:
        """Plot the clusters of the given observation. It only plots the first three
        dimensions.

        Arguments:
        ---------
            example_name (str):
                The name of the example.
        """

        print("Computing KNN Clusters")
        fig = go.Figure()
        for i in range(self.data.shape[0]):
            _, nearest_data = self.predict(i)
            # Plot the clusters
            fig.add_trace(
                go.Scatter3d(
                    x=nearest_data[:, 0],
                    y=nearest_data[:, 1],
                    z=nearest_data[:, 2],
                    mode="markers",
                    marker=dict(
                        size=3,
                        opacity=0.5,
                        color=i,
                    ),
                )
            )
        print(f"Saving results as knn_clusters_{example_name}.html")
        # Save to html
        fig.write_html(f"results/knn_clusters_{example_name}.html")

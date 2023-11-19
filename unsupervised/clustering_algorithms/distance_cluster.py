"""A module for clustering using distances."""

from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from unsupervised.distances import lp_distance
from unsupervised.utils import pairwise_distance


class DistanceCluster:
    def __init__(
        self,
        data: pd.DataFrame | np.ndarray,
        n_clusters: int,
        distance: Callable = lp_distance,
        distance_kwargs: dict | None = None,
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
        """

        self.data = np.array(data)
        self.n_clusters = n_clusters
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
        # Compute thresholds
        thresholds = max(distances) / self.n_clusters
        clusters = [[]] * self.n_clusters
        # Make predictions
        for i in range(self.n_clusters):
            less_than = distances <= thresholds * (i + 1)
            greater_than = distances >= thresholds * i
            clusters[i] = np.argwhere(less_than & greater_than).flatten()

        return clusters

    def plot_clusters(self, example_name: str, obs_index: int) -> None:
        """Plot the clusters of the given observation. It only plots the first three
        dimensions.

        Arguments:
        ---------
            example_name (str):
                The name of the example.
        """

        example_name = example_name + f"{self.data.shape[1]}_dim"
        fig = go.Figure()
        clusters = self.predict(obs_index)
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
        fig.write_html(f"results/distance_clusters_{example_name }.html")

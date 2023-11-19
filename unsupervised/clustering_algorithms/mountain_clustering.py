"""A module for clustering using mountain clustering algorithm."""

from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from unsupervised.distances import lp_distance
from unsupervised.utils import create_nd_grid, pairwise_distance


class MountainClustering:
    def __init__(
        self,
        data: pd.DataFrame | np.ndarray,
        num_grid_partitions: int,
        distance: Callable = lp_distance,
        distance_kwargs: dict | None = None,
        sigma: float = 1,
        beta: float = 1,
        max_iterations: int = 100,
    ):
        """Initialise the model.

        Arguments:
        ---------
            data (pd.DataFrame):
                The data to cluster.
            distance (Callable, optional):
                The distance function to use. Defaults to lp_distance.
            distance_kwargs (dict, optional):
                The distance function keyword arguments. Defaults to None.
            sigma (float, optional):
                The sigma parameter. Defaults to 1.
            beta (float, optional):
                The beta parameter. Defaults to 1.
            max_iterations (int, optional):
                The maximum number of iterations. Defaults to 100.
        """

        self.data = np.array(data)
        self.grid = create_nd_grid(num_grid_partitions, self.data.shape[1])
        self.distances = pairwise_distance(
            self.grid, self.data, distance, distance_kwargs
        )
        self.grid_distances = pairwise_distance(
            self.grid, self.grid, distance, distance_kwargs
        )
        self.sigma = sigma
        self.beta = beta
        self.max_iterations = max_iterations

    def _mountain_function_0(self) -> float:
        """Compute the mountain function for a given point.

        Returns:
        -------
            float:
                The distance between the two points.
        """

        result = np.sum(np.exp(-self.distances**2 / (2 * self.sigma**2)), axis=1)
        return result

    def _mountain_function(
        self, center_index: int, prev_mountain_function: list
    ) -> float:
        """Compute the mountain function for a given point.

        Arguments:
        ---------
            center_index (int):
                The index of the center.
            prev_mountain_function (list):
                The previous result of the mountain function computation

        Returns:
        -------
            float:
                The mountain_function.
        """

        return prev_mountain_function - prev_mountain_function[center_index] * np.exp(
            -self.grid_distances[center_index] ** 2 / (2 * self.beta**2)
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

        previous_centers = set()

        # Compute the first center
        mountain_function = self._mountain_function_0()
        center_index = np.argmax(mountain_function)

        for _ in range(self.max_iterations):
            if center_index in previous_centers:
                break
            mountain_function = self._mountain_function(center_index, mountain_function)
            previous_centers.add(center_index)
            center_index = np.argmax(mountain_function)
        previous_centers.add(center_index)
        centers = np.array(list(previous_centers))
        return self.grid[centers]

    def plot_clusters(self, example_name: str) -> None:
        """Plot the clusters of the given observation. It only plots the first three
        dimensions.

        Arguments:
        ---------
            example_name (str):
                The name of the example.
        """

        example_name = example_name + f"{self.data.shape[1]}_dim"
        fig = go.Figure()
        clusters = self.predict()
        fig.add_trace(
            go.Scatter3d(
                x=self.grid[clusters, 0],
                y=self.grid[clusters, 1],
                z=self.grid[clusters, 2],
                mode="markers",
                marker=dict(
                    size=3,
                    opacity=0.5,
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
        fig.write_html(f"results/mountain_clusters_{example_name }.html")

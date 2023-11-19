"""A clustering algorithm based on the connected components of a distance matrix.
Two points are connected if their distance is less than a threshold."""

from collections import deque
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from unsupervised.distances import lp_distance
from unsupervised.utils import pairwise_distance


class ConnectedComponentsCluster:
    def __init__(
        self,
        data: pd.DataFrame | np.ndarray,
        threshold: float,
        distance: Callable = lp_distance,
        distance_kwargs: dict | None = None,
    ):
        """Initialise the model.

        Arguments:
        ---------
            data (pd.DataFrame):
                The data to cluster.
            threshold (float):
                The threshold to use to connect two points.
            distance (Callable, optional):
                The distance function to use. Defaults to lp_distance.
            distance_kwargs (dict, optional):
                The distance function keyword arguments. Defaults to None.
        """

        self.data = data
        self.threshold = threshold
        self.distance_matrix = pairwise_distance(
            self.data, self.data, distance, distance_kwargs
        )

    def get_connected_components(self, threshold: float) -> list:
        """Get the connected components of the distance matrix.

        Arguments:
        ---------
            threshold (float):
                The threshold to use to connect two points.

        Returns:
        -------
            list:
                The connected components.
        """

        # Get the indices of the points that are connected
        indices = np.argwhere(self.distance_matrix < threshold)
        seen = set()
        components = []
        for root in range(self.data.shape[0]):
            if root not in seen:
                seen.add(root)
                component = []
                queue = deque([root])

                while queue:
                    node = queue.popleft()
                    component.append(node)
                    for neighbor in indices[indices[:, 0] == node][:, 1]:
                        if neighbor not in seen:
                            seen.add(neighbor)
                            queue.append(neighbor)
                components.append(component)
        return components

    def predict(self) -> np.ndarray:
        """Predict the clusters.

        Returns:
        -------
            np.ndarray:
                The predicted clusters.
        """

        # Get the connected components
        clusters = self.get_connected_components(self.threshold)
        return clusters

    def plot_clusters(self, example_name: str) -> None:
        """Plot the clusters. It only plots the first three dimensions.

        Arguments:
        ---------
            example_name (str):
                The name of the example.
        """

        example_name = example_name + f"{self.data.shape[1]}_dim"
        fig = go.Figure()
        clusters = self.predict()

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
        fig.write_html(f"results/connected_clusters_{example_name }.html")

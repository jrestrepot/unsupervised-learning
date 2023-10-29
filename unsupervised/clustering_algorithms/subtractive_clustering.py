"""A module for clustering using mountain clustering algorithm."""

from typing import Callable
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from unsupervised.distances import lp_distance
from unsupervised.utils import pairwise_distance


class SubtractiveClustering:
    def __init__(
        self,
        data: pd.DataFrame | np.ndarray,
        distance: Callable = lp_distance,
        distance_kwargs: dict | None = None,
        ra: float = 1,
        rb: float = 1,
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
            ra (float, optional):
                The ra parameter. Defaults to 1.
            rb (float, optional):
                The rb parameter. Defaults to 1.
            max_iterations (int, optional):
                The maximum number of iterations. Defaults to 100.
        """

        self.data = np.array(data)
        self.distances = pairwise_distance(
            self.data, self.data, distance, distance_kwargs
        )
        self.ra = ra
        self.rb = rb
        self.max_iterations = max_iterations

    def density_measure_0(self) -> float:
        """Compute the first densities.

        Returns:
        -------
            float:
                The densities for all the data.
        """

        result = np.sum(np.exp(-self.distances ** 2 / (self.ra / 2) ** 2), axis = 0)
        return result

    def density_measure(
        self, center_index: int, prev_densities: list
    ) -> float:
        """Compute the densities for a given point.

        Arguments:
        ---------
            center_index (int):
                The index of the center.
            prev_densities (list):
                The previous result of the densities computation

        Returns:
        -------
            float:
                The new densities
        """

        func_center = prev_densities[center_index]

        return prev_densities - func_center * np.exp(
            -self.distances[center_index] ** 2
            / (self.rb / 2) ** 2
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

        print(f"Computing Subtractive Clustering...")
        previous_centers = set()

        # Compute the first center
        densities = self.density_measure_0()
        center_index = np.argmax(densities)

        for _ in range(self.max_iterations):
            if center_index in previous_centers:
                break
            densities = self.density_measure(center_index, densities)
            previous_centers.add(center_index)
            center_index = np.argmax(densities)
        previous_centers.add(center_index)
        return np.array(list(previous_centers))

    def plot_clusters(self, example_name: str) -> None:
        """Plot the clusters of the given observation. It only plots the first three
        dimensions.

        Arguments:
        ---------
            example_name (str):
                The name of the example.
        """

        fig = go.Figure()
        clusters = self.predict()
        print(f"Saving results as subtractive_centers_{example_name}.html")
        fig.add_trace(
            go.Scatter3d(
                x=self.data[clusters, 0],
                y=self.data[clusters, 1],
                z=self.data[clusters, 2],
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
        # Save to html
        fig.write_html(f"results/subtractive_centers_{example_name}.html")

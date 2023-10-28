"""A module for clustering using mountain clustering algorithm."""

from typing import Callable
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from unsupervised.distances import lp_distance
from unsupervised.utils import create_nd_grid


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
        self.distance = distance
        self.distance_kwargs = distance_kwargs
        self.grid = create_nd_grid(num_grid_partitions, self.data.shape[1])
        self.sigma = sigma
        self.beta = beta
        self.max_iterations = max_iterations

    def mountain_function_0(self, point: np.ndarray) -> float:
        """Compute the mountain function for a given point.

        Arguments:
        ---------
            point (np.ndarray):
                The point in the grid that we want to evaluate.
            data (np.ndarray):
                The dataset.

        Returns:
        -------
            float:
                The distance between the two points.
        """

        result = 0
        for i in range(self.data.shape[0]):
            result += np.exp(
                -self.distance(point, self.data[i], **self.distance_kwargs) ** 2
                / (2 * self.sigma**2)
            )
        return result

    def mountain_function(
        self, point_index: int, center_index: int, prev_mountain_function: list
    ) -> float:
        """Compute the mountain function for a given point.

        Arguments:
        ---------
            point_index (int):
                The index of the point in the grid that we want to evaluate.
            center_index (int):
                The index of the center.
            prev_mountain_function (list):
                The previous result of the mountain function computation

        Returns:
        -------
            float:
                The mountain_function.
        """

        center = self.grid[center_index]
        point = self.grid[point_index]
        func_center = prev_mountain_function[center_index]
        func_point = prev_mountain_function[point_index]

        return func_point - func_center * np.exp(
            -self.distance(point, center, **self.distance_kwargs) ** 2
            / (2 * self.beta**2)
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

        print(f"Computing Mountain Clustering...")
        try:
            cpus = cpu_count()
        except NotImplementedError:
            cpus = 2  # arbitrary default

        previous_centers = set()

        # Compute the first center
        pool = Pool(processes=cpus)
        mountain_function = pool.map(self.mountain_function_0, self.grid)
        center_index = np.argmax(mountain_function)
        
        for _ in range(self.max_iterations):
            if center_index in previous_centers:
                break
            mountain_function = pool.starmap(
                self.mountain_function,
                [(i, center_index, mountain_function) for i in range(len(self.grid))],
            )
            previous_centers.add(center_index)
            center_index = np.argmax(mountain_function)
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
        print(f"Saving results as distance_clusters_{example_name}.html")
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
        # Save to html
        fig.write_html(f"results/mountain_centers_{example_name}.html")

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
            clusters[i] = self.data[distances <= thresholds * (i + 1)]

        return clusters

"""A module for clustering using kmeans algorithm."""

from typing import Callable
from multiprocessing import Pool, cpu_count
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
        centers: np.ndarray | None = None,
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
            centers (np.ndarray, optional):
                The initial centers. Defaults to None.
        """

        self.data = np.array(data)
        self.distance = distance
        self.distance_kwargs = distance_kwargs
        self.n_clusters = n_clusters
        self.centers = centers
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
        distance_matrix = pairwise_distance(self.centers, self.data, self.distance, self.distance_kwargs)
        # Getting the min distance is another way of viewing the function uij
        minimum_distances_index = np.argmin(distance_matrix, axis = 0)
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

        return np.sum(membership_matrix*distance_matrix**2)
        
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
            self.centers[i] = 1/np.sum(membership_matrix[i])*np.sum(membership_matrix[i]*self.data.T, axis = 1)


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

        print(f"Computing KMeans...")
        previous_loss = np.inf
        # Initiliaze centers randomly if not given
        if self.centers is None:
            self.centers = self.data[np.random.choice(self.data.shape[0], self.n_clusters, replace = False)]
        for _ in range(self.max_iterations):
            membership_matrix, distance_matrix = self.get_membership_matrix()
            self.update_centers(membership_matrix)
            loss = self.loss_function(membership_matrix, distance_matrix)
            if abs(previous_loss - loss) < self.tolerance:
                break
            previous_loss = loss
        assignation = np.argmax(membership_matrix, axis = 0)
        clusters = [np.where(assignation == i)[0] for i in range(self.n_clusters)]
        return clusters



    def plot_clusters(self, example_name: str) -> None:
        """Plot the clusters. It only plots the first three dimensions.

        Arguments:
        ---------
            example_name (str):
                The name of the example.
        """

        fig = go.Figure()
        clusters = self.predict()

        print(f"Saving results as kmeans_clusters_{example_name}.html")
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
        # Save to html
        fig.write_html(f"results/kmeans_clusters_{example_name}.html")



# class KMeans():
#   def __init__(self, n_clusters, kind, inv_cov_matrix = None):
#     self.n_clusters = n_clusters
#     self.kind = kind
#     self.losses = []
#     self.inv_cov_matrix = inv_cov_matrix

  
#   def compute_loss(self, X, membership_matrix, distance_matrix, d):
#     loss = 0
#     intra_cluster_vectors_sums = np.zeros((self.n_clusters,d))
#     count = 1
#     for i in range(self.n_clusters):
#       aux_indexes = np.where(membership_matrix[i,:] == 1)[0]
#       intra_cluster_vectors_sums[i,:] = np.sum(X[aux_indexes, :], axis = 0)
#       intra_cluster_sum = np.sum(distance_matrix[i,aux_indexes])
#       loss += intra_cluster_sum
#       count +=1
#     return loss, intra_cluster_vectors_sums

#   def train(self, X, tolerance, loss_tolerance, verbose = False):
#     n_points, d = X.shape
#     initial_indexes = np.random.choice(n_points, self.n_clusters, replace = False)
#     initial_centers = X[initial_indexes]
#     initial_membership_matrix,initial_distance_matrix = self.assign_points(X, initial_centers, n_points)
#     initial_loss, intra_cluster_vectors_sums = self.compute_loss(X, initial_membership_matrix,initial_distance_matrix,d)
#     self.losses.append(initial_loss)
#     criteria = initial_loss
#     membership_matrix = initial_membership_matrix
#     count = 1
#     while (criteria > tolerance) & (self.losses[-1] > loss_tolerance):
#       centers_cardinality = np.sum(membership_matrix, axis = 1)
#       new_centers = intra_cluster_vectors_sums/centers_cardinality.reshape(-1,1)
#       membership_matrix, distance_matrix = self.assign_points(X, new_centers, n_points)
#       loss, intra_cluster_vectors_sums = self.compute_loss(X,membership_matrix,distance_matrix,d)
#       if verbose:
#         print(f"iteration {count}: {loss}")
#       self.losses.append(loss)
#       criteria = abs(self.losses[count-1] - self.losses[count])
#       count += 1
#     if verbose:
#       print(f"final loss: {self.losses[-1]}")
#     centers = new_centers
#     return centers, membership_matrix
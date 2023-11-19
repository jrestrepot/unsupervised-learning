"""A module for descriptive statistics."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from umap import UMAP


def descriptive_statistics(data: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """A function to compute descriptive statistics.

    Arguments:
    ---------
        data (pd.DataFrame | np.ndarray):
            The data to compute the descriptive statistics of.

    Returns:
    -------
        pd.DataFrame:
            The descriptive statistics.
    """

    if isinstance(data, pd.DataFrame):
        return data.describe()
    elif isinstance(data, np.ndarray):
        return pd.DataFrame(data).describe()
    else:
        raise ValueError("Input data should be a pandas DataFrame or numpy ndarray.")


def plot_histogram(data: pd.DataFrame | np.ndarray, column: str) -> None:
    """A function to plot a histogram.

    Arguments:
    ---------
        data (pd.DataFrame | np.ndarray):
            The data to plot the histogram of.
        column (str):
            The column to plot the histogram of.
    """

    if isinstance(data, pd.DataFrame):
        fig = px.histogram(data, x=column, title=f"Histogram of {column}")
        # Save to png
        fig.write_html(f"results/histogram_{column }.html")
    elif isinstance(data, np.ndarray):
        fig = go.Figure(data=[go.Histogram(x=data, name=column)])
        fig.update_layout(title=f"Histogram of {column}", xaxis_title=column)
        # Save to png
        fig.write_html(f"results/histogram_{column }.html")
    else:
        raise ValueError("Input data should be a pandas DataFrame or numpy ndarray.")


def plot_boxplot(data: pd.DataFrame | np.ndarray, column: str) -> None:
    """A function to plot a boxplot.

    Arguments:
    ---------
        data (pd.DataFrame | np.ndarray):
            The data to plot the boxplot of.
        column (str):
            The column to plot the boxplot of.
    """

    if isinstance(data, pd.DataFrame):
        fig = px.box(data, y=column, title=f"Boxplot of {column}")
        # Save to png
        fig.write_html(f"results/boxplot_{column }.html")
    elif isinstance(data, np.ndarray):
        fig = go.Figure(data=[go.Box(y=data, name=column)])
        fig.update_layout(title=f"Boxplot of {column}", yaxis_title=column)
        # Save to png
        fig.write_html(f"results/boxplot_{column }.html")
    else:
        raise ValueError("Input data should be a pandas DataFrame or numpy ndarray.")


def biplot(data: pd.DataFrame | np.ndarray) -> None:
    """A function to plot the first two principal components of the data.

    Arguments:
    ---------
        data (pd.DataFrame | np.ndarray):
            The data to plot the PCA of.
    """

    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)

    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)

    # Get the PCA loadings (coefficients)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    # Create a DataFrame for loadings
    loadings_df = pd.DataFrame(loadings, columns=["PC1", "PC2"], index=data.columns)

    # Create a biplot
    fig = px.scatter(pca_result, x=0, y=1, title="PCA Biplot")

    # Add arrows for variable loadings
    for feature in loadings_df.index:
        x, y = loadings_df.loc[feature, "PC1"], loadings_df.loc[feature, "PC2"]
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=x,
            y1=y,
            line=dict(color="red", width=2),
            xref="x",
            yref="y",
        )
        fig.add_annotation(
            x=x,
            y=y,
            text=feature,
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="red",
        )

    # Customize the scatter plot appearance
    fig.update_traces(marker=dict(size=12, opacity=0.8))
    fig.update_xaxes(title_text="Principal Component 1")
    fig.update_yaxes(title_text="Principal Component 2")

    # Save to png
    fig.write_html("results/biplot.html")


def pairplot(data: pd.DataFrame | np.ndarray, hue: str) -> None:
    """A function to plot a pairplot.

    Arguments:
    ---------
        data (pd.DataFrame | np.ndarray):
            The data to plot the pairplot of.
        hue (str):
            The column to use for coloring the data.
    """

    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    dimensions = data.drop(hue, axis=1).columns
    fig = ff.create_scatterplotmatrix(
        data, diag="histogram", index=hue, width=1500, height=700
    )
    # Save to png
    fig.write_html("results/pairplot.html")


def umap(data: pd.DataFrame | np.ndarray, target_column: str | None = None) -> None:
    """A function to plot the UMAP of the data.

    Arguments:
    ---------
        data (pd.DataFrame | np.ndarray):
            The data to plot the UMAP of.
        target_column (str):
            The column to use for coloring the data.
    """

    reducer = UMAP(random_state=42, n_jobs=-1)
    embedding = reducer.fit_transform(data.drop(columns=[target_column]))
    fig = px.scatter(embedding, x=0, y=1, title="UMAP")
    fig.write_html("results/umap.html")
    embedding = pd.DataFrame(embedding)
    embedding[target_column] = data[target_column]
    fig = px.scatter(embedding, x=0, y=1, title="UMAP", color=target_column)
    fig.write_html(f"results/umap_{target_column }.html")
    return embedding.drop(columns=[target_column])

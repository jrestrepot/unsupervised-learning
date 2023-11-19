import numpy as np
import plotly.graph_objects as go

from unsupervised.tests import (
    test_high_dimensions,
    test_low_dimensions,
    test_original_dimensions,
)
from unsupervised.utils import analyze_and_transform_data, get_params, read_data

PARAMETERS_FILE = "parameters.json"

# Set the parameters
(
    DATA_PATH,
    DISTANCE,
    DISTANCE_KWARGS,
    DISTANCE_THRESHOLD,
    K,
    N_CLUSTERS_KMEANS,
    N_CLUSTERS_FUZZY_CMEANS,
    M_FUZZY_CMEANS,
    N_CLUSTERS_DISTANCE,
    NUM_PARTITIONS,
    SIGMA,
    BETA,
    RA,
) = get_params(PARAMETERS_FILE)


def main():
    # Seed for consistency
    np.random.seed(42)
    # Read data
    data = read_data(DATA_PATH)
    # Drop the id column
    data.drop(["Id"], axis=1, inplace=True)
    data, umap_embedding = analyze_and_transform_data(data, "Species")
    # Define RB as 1.5 * RA
    RB = 1.5 * np.array(RA)

    # # Test with different dimensions
    test_original_dimensions(
        data,
        DISTANCE,
        DISTANCE_KWARGS,
        DISTANCE_THRESHOLD,
        M_FUZZY_CMEANS,
        N_CLUSTERS_DISTANCE,
        NUM_PARTITIONS,
        SIGMA,
        BETA,
        RA,
        RB,
    )
    test_low_dimensions(
        data,
        DISTANCE,
        DISTANCE_KWARGS,
        DISTANCE_THRESHOLD,
        M_FUZZY_CMEANS,
        N_CLUSTERS_DISTANCE,
        NUM_PARTITIONS,
        SIGMA,
        BETA,
        RA,
        RB,
    )
    test_high_dimensions(
        data,
        DISTANCE,
        DISTANCE_KWARGS,
        DISTANCE_THRESHOLD,
        M_FUZZY_CMEANS,
        N_CLUSTERS_DISTANCE,
        NUM_PARTITIONS,
        SIGMA,
        BETA,
        RA,
        RB,
    )
    test_original_dimensions(
        umap_embedding,
        DISTANCE,
        DISTANCE_KWARGS,
        DISTANCE_THRESHOLD,
        M_FUZZY_CMEANS,
        N_CLUSTERS_DISTANCE,
        NUM_PARTITIONS,
        SIGMA,
        BETA,
        RA,
        RB,
    )


if __name__ == "__main__":
    main()

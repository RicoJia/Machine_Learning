import numpy as np

from .k_nearest_neighbor import KNearestNeighbor


def collaborative_filtering(
    input_array, n_neighbors, distance_measure="euclidean", aggregator="mode"
):
    """
    This is a wrapper function for your KNearestNeighbors class, that runs kNN
    as a collaborative filter.

    If there is a 0 in the array, you must impute a value determined by using your
    kNN classifier as a collaborative filter. All non-zero entries should remain
    the same.

    For example, if `input_array`(containing data we are trying to impute) looks like:

        [[0, 2],
         [1, 2],
         [1, 0]]

    We are trying to impute the 0 values by replacing the 0 values with an aggregation of the
    neighbors for that row. The features that are 0 in the row are replaced with an aggregation
    of the corresponding column of the neighbors of that row. For example, if aggregation is 'mean',
    and K = 2 then the output should be:

        [[1, 2],
         [1, 2],
         [1, 2]]

    Note that the row you are trying to impute for is ignored in the aggregation.
    Use `ignore_first = True` in the predict function of the KNN to accomplish this. If
    `ignore_first = False` and K = 2, then the result would be:

        [[(1 + 0) / 2 = .5, 2],
         [1, 2],
         [1, (2 + 0) / 2 = 1]]

        = [[.5, 2],
           [1, 2],
           [1, 1]]

    This is incorrect because the value that we are trying to replace is considered in the
    aggregation.

    The non-zero values are left untouched. If aggregation is 'mode', then the output should be:

        [[1, 2],
         [1, 2],
         [1, 2]]


    Arguments:
        input_array {np.ndarray} -- An input array of shape (n_samples, n_features).
            Any zeros will get imputed.
        n_neighbors {int} -- Number of neighbors to use for prediction.
        distance_measure {str} -- Which distance measure to use. Can be one of
            'euclidean', 'manhattan', or 'cosine'. This is the distance measure
            that will be used to compare features to produce labels.
        aggregator {str} -- How to aggregate a label across the `n_neighbors` nearest
            neighbors. Can be one of 'mode', 'mean', or 'median'.

    Returns:
        imputed_array {np.ndarray} -- An array of shape (n_samples, n_features) with imputed
            values for any zeros in the original input_array.
    """

    impute_array = np.zeros(input_array.shape[1])
    cp_input_array = np.copy(input_array)
    cp2_input_array = np.copy(input_array)
    for cln_num in range(cp_input_array.shape[1]):
        sorted_cln = cp2_input_array[:, cln_num]
        sorted_cln.sort()
        first_non_zero_index = (sorted_cln == 0).sum()
        k_neighbors = sorted_cln[
            first_non_zero_index : first_non_zero_index + n_neighbors
        ]
        k_neighbors_ls = k_neighbors.tolist()
        if aggregator == "mode":
            impute_array[cln_num] = max(set(k_neighbors_ls), key=k_neighbors_ls.count)
        elif aggregator == "mean":
            impute_array[cln_num] = k_neighbors.sum() / k_neighbors.shape[0]
        elif aggregator == "median":
            impute_array[cln_num] = k_neighbors_ls[int(k_neighbors.shape[0] / 2)]

        for row_num in range(input_array.shape[0]):
            if cp_input_array[row_num][cln_num] == 0:
                cp_input_array[row_num][cln_num] = impute_array[cln_num]

    return cp_input_array

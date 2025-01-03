import numpy as np

from .distances import cosine_distances, euclidean_distances, manhattan_distances


class KNearestNeighbor:
    def __init__(self, n_neighbors, distance_measure="euclidean", aggregator="mode"):
        """
        K-Nearest Neighbor is a straightforward algorithm that can be highly
        effective. Training time is...well...is there any training? At test time, labels for
        new points are predicted by comparing them to the nearest neighbors in the
        training data.

        ```distance_measure``` lets you switch between which distance measure you will
        use to compare data points. The behavior is as follows:

        If 'euclidean', use euclidean_distances, if 'manhattan', use manhattan_distances,
        if  'cosine', use cosine_distances.

        ```aggregator``` lets you alter how a label is predicted for a data point based
        on its neighbors. If it's set to `mean`, it is the mean of the labels of the
        neighbors. If it's set to `mode`, it is the mode of the labels of the neighbors.
        If it is set to median, it is the median of the labels of the neighbors. If the
        number of dimensions returned in the label is more than 1, the aggregator is
        applied to each dimension independently. For example, if the labels of 3
        closest neighbors are:
            [
                [1, 2, 3],
                [2, 3, 4],
                [3, 4, 5]
            ]
        And the aggregator is 'mean', applied along each dimension, this will return for
        that point:
            [
                [2, 3, 4]
            ]

        Arguments:
            n_neighbors {int} -- Number of neighbors to use for prediction.
            distance_measure {str} -- Which distance measure to use. Can be one of
                'euclidean', 'manhattan', or 'cosine'. This is the distance measure
                that will be used to compare features to produce labels.
            aggregator {str} -- How to aggregate a label across the `n_neighbors` nearest
                neighbors. Can be one of 'mode', 'mean', or 'median'.
        """
        self.n_neighbors = n_neighbors
        self.model_features = None
        self.model_targets = None
        self.aggregator = aggregator
        self.distance_measure = distance_measure

    def fit(self, features, targets):
        """Fit features, a numpy array of size (n_samples, n_features). For a KNN, this
        function should store the features and corresponding targets in class
        variables that can be accessed in the `predict` function. Note that targets can
        be multidimensional!

        HINT: One use case of KNN is for imputation, where the features and the targets
        are the same. See tests/test_collaborative_filtering for an example of this.

        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            targets {[type]} -- Target labels for each data point, shape of (n_samples,
                n_dimensions).
        """
        self.model_features = features
        self.model_targets = targets

    def predict(self, features, ignore_first=False):
        """Predict from features, a numpy array of size (n_samples, n_features) Use the
        training data to predict labels on the test features. For each testing sample, compare it
        to the training samples. Look at the self.n_neighbors closest samples to the
        test sample by comparing their feature vectors. The label for the test sample
        is the determined by aggregating the K nearest neighbors in the training data.

        Note that when using KNN for imputation, the predicted labels are the imputed testing data
        and the shape is (n_samples, n_features).

        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            ignore_first {bool} -- If this is True, then we ignore the closest point
                when doing the aggregation. This is used for collaborative
                filtering, where the closest point is itself and thus is not a neighbor.
                In this case, we would use 1:(n_neighbors + 1).

        Returns:
            labels {np.ndarray} -- Labels for each data point, of shape (n_samples,
                n_dimensions). This n_dimensions should be the same as n_dimensions of targets in fit function.
        """
        feature_cln = self.model_features.shape[1]
        targets_cln = self.model_targets.shape[1]
        predicted_targets = np.empty((0, targets_cln))

        if self.distance_measure == "manhattan":
            dist_matrix = manhattan_distances(features, self.model_features)
        elif self.distance_measure == "cosine":
            dist_matrix = cosine_distances(features, self.model_features)
        elif self.distance_measure == "euclidean":
            dist_matrix = euclidean_distances(features, self.model_features)

        for row in dist_matrix:
            # print("----row shape: ", row.shape)
            # print("model shape: ", self.model_features.shape[0])
            total_matrix = np.hstack(
                (self.model_features, self.model_targets, row.reshape(row.shape[0], 1))
            )
            # sort the matrix by distance
            total_matrix = total_matrix[total_matrix[:, -1].argsort()]
            if ignore_first == True:
                first_non_zero_index = (total_matrix[:, -1] == 0).sum()
                total_matrix = total_matrix[first_non_zero_index:]

            # get k neighbors
            k_neighbors = total_matrix[
                : self.n_neighbors, feature_cln : feature_cln + targets_cln
            ]
            k_neighbors_ls = k_neighbors.tolist()

            # make prediction
            if (
                self.aggregator == "mode"
            ):  # Assumption: all these functions are applied along each column
                prediction = np.apply_along_axis(
                    lambda x: max(set(x.tolist()), key=(x.tolist()).count),
                    axis=0,
                    arr=k_neighbors,
                )
            elif self.aggregator == "mean":
                prediction = np.mean(k_neighbors, axis=0)
            elif self.aggregator == "median":
                prediction = np.median(k_neighbors, axis=0)

            predicted_targets = np.vstack((predicted_targets, prediction))

        return predicted_targets

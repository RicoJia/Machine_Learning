import os
import random
from itertools import permutations

import numpy as np
from scipy.stats import multivariate_normal
from src import GMM, KMeans, adjusted_mutual_info, generate_cluster_data

print("Starting example experiment")


def _test_gmm_parameters(covariance_type):
    n_samples = [1000]
    n_centers = [2]
    stds = [0.1, 0.5]
    n_features = [2, 4]

    for n in n_samples:
        for f in n_features:
            for c in n_centers:
                for s in stds:
                    features, targets = generate_cluster_data(
                        n_samples=n, n_features=f, n_centers=c, cluster_stds=s
                    )
                    # make model and fit
                    model = GMM(c, covariance_type=covariance_type)
                    model.fit(features)
                    covariances = model.covariances
                    for cov in covariances:
                        print("mean cov: ", np.abs(np.sqrt(cov) - s).mean())
                        if np.abs(np.sqrt(cov) - s).mean() < 1e-1:
                            return
                    #
                    #
                    # means = model.means
                    # orderings = permutations(means)
                    # distance_to_true_means = []
                    #
                    # actual_means = np.array([
                    #     features[targets == i, :].mean(axis=0) for i in range(targets.max() + 1)
                    # ])
                    #
                    # for ordering in orderings:
                    #     _means = np.array(list(ordering))
                    #
                    #     distance_to_true_means.append(
                    #         np.abs(_means - actual_means).sum()
                    #     )
                    # assert (min(distance_to_true_means) < 1e-1)
                    #
                    # mixing_weights = model.mixing_weights
                    # orderings = permutations(mixing_weights)
                    # distance_to_true_mixing_weights = []
                    #
                    # actual_mixing_weights = np.array([
                    #     features[targets == i, :].shape[0] for i in range(targets.max() + 1)
                    # ])
                    # actual_mixing_weights = actual_mixing_weights / actual_mixing_weights.sum()
                    #
                    # for ordering in orderings:
                    #     _mixing_weights = np.array(list(ordering))
                    #     distance_to_true_mixing_weights.append(
                    #         np.abs(_mixing_weights - actual_mixing_weights).sum()
                    #     )
                    # assert (min(distance_to_true_mixing_weights) < 1e-1)
                    #
                    # # predict and calculate adjusted mutual info
                    # labels = model.predict(features)
                    # acc = adjusted_mutual_info(targets, labels)
                    # assert (acc >= .9)


_test_gmm_parameters("spherical")

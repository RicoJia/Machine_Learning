import numpy as np
from src import KMeans
from scipy.stats import multivariate_normal

class GMM():
    def __init__(self, n_clusters, covariance_type):
        """
        This class implements a Gaussian Mixture Model updated using expectation
        maximization.

        A useful tutorial:
            http://cal.cs.illinois.edu/~johannes/research/EM%20derivations.pdf

        The EM algorithm for GMMs has two steps:

        1. Update posteriors (assignments to each Gaussian)
        2. Update Gaussian parameters (means, variances, and priors for each Gaussian)

        While you only have to implement the fit and predict functions to pass the
        test cases, we recommend that you break these two steps apart into separate
        functions. We have provided a template for you to put your code in.

        Use only numpy to implement this algorithm.

        This function MUST, after running 'fit', have variables named 'means' and
        'covariances' in order to pass the test cases. These variables are checked by the
        test cases to make sure you have recovered cluster parameters accurately.

        The fit and predict functions are implemented for you. To complete the implementation,
        you must implement:
            - _e_step
            - _m_step
            - _log_likelihood

        Args:
            n_clusters (int): Number of Gaussians to cluster the given data into.
            covariance_type (str): Either 'spherical', 'diagonal'. Determines the
                covariance type for the Gaussians in the mixture model.

        """
        self.n_clusters = n_clusters
        allowed_covariance_types = ['spherical', 'diagonal']
        # if covariance_type not in allowed_covariance_types:
        #     raise ValueError(f'covariance_type must be in {allowed_covariance_types}')
        self.covariance_type = covariance_type

        self.means = None
        self.covariances = None
        self.mixing_weights = None
        self.max_iterations = 200

    def fit(self, features):
        """
        Fit GMM to the given data using `self.n_clusters` number of Gaussians.
        Features can have greater than 2 dimensions.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            None (saves model - means, covariances, and mixing weights - internally)
        """
        # 1. Use your KMeans implementation to initialize the means of the GMM.
        kmeans = KMeans(self.n_clusters)
        kmeans.fit(features)
        self.means = kmeans.means

        # 2. Initialize the covariance matrix and the mixing weights
        self.covariances = self._init_covariance(features.shape[-1])

        # 3. Initialize the mixing weights
        self.mixing_weights = np.random.rand(self.n_clusters)
        self.mixing_weights /= np.sum(self.mixing_weights)

        # 4. Compute log_likelihood under initial random covariance and KMeans means.
        prev_log_likelihood = -float('inf')
        log_likelihood = self._overall_log_likelihood(features)

        # 5. While the log_likelihood is increasing significantly, or max_iterations has
        # not been reached, continue EM until convergence.
        n_iter = 0
        while log_likelihood - prev_log_likelihood > 1e-4 and n_iter < self.max_iterations:
            prev_log_likelihood = log_likelihood

            assignments = self._e_step(features)
            self.means, self.covariances, self.mixing_weights = (
                self._m_step(features, assignments)
            )
            log_likelihood = self._overall_log_likelihood(features)
            # print("----------------")
            # print("log likelihood: ", log_likelihood)
            n_iter += 1

    def predict(self, features):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict the label
        of each sample (e.g. the index of the Gaussian with the highest posterior for that
        sample).

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted assigment to each cluster for each sample,
                of size (n_samples,). Each element is which cluster that sample belongs to.
        """
        posteriors = self._e_step(features).T
        return np.argmax(posteriors, axis=-1)

    def _e_step(self, features):
        """
        The expectation step in Expectation-Maximization. Given the current class member
        variables self.mean, self.covariance, and self.mixing_weights:
            1. Calculate the log_likelihood of each point under each Gaussian.
            2. Calculate the posterior probability for each point under each Gaussian
            3. Return the posterior probability (assignments).
        
        This function should call your implementation of _log_likelihood (which should call
        multvariate_normal.logpdf). This should use the Gaussian parameter contained in
        self.means, self.covariance, and self.mixing_weights

        Arguments:
            features {np.ndarray} -- Features to apply means, covariance, and mixing_weights
                to.

        Returns:
            np.ndarray -- Posterior probabilities to each Gaussian (shape is
                ( self.n_clusters, features.shape[0]))
        """
        all_posteriors = np.zeros((self.n_clusters, features.shape[0]))

        #calculate posteriors for all points, all gaussians
        for i_feature in range(features.shape[0]):
            for k_idx in range(self.n_clusters):
                numerator = self._log_likelihood(features[i_feature, :].reshape(1, features.shape[1]), k_idx) #number

                #sum of log probability of the same feature, all gaussians.
                sum_prob = 0.0
                for k_idx_2 in range(self.n_clusters):
                    sum_prob += self._log_likelihood(features[i_feature, :].reshape(1, features.shape[1]), k_idx_2)
                #gamma_jn = numerator/denom
                all_posteriors[k_idx, i_feature] = numerator/sum_prob

            # log_likelihood_vec = self._log_likelihood(features, k_idx)
            # all_posteriors[:, k_idx] = log_likelihood_vec/np.sum(log_likelihood_vec)


        for k_idx in range(self.n_clusters):
            all_posteriors[k_idx, :] = self._posterior(features, k_idx)
        return all_posteriors


    def _m_step(self, features, assignments):
        """
        Maximization step in Expectation-Maximization. Given the current features and
        assignments, update self.means, self.covariances, and self.mixing_weights. Here,
        you implement the update equations for the means, covariances, and mixing weights.
            1. Update the means with the mu_j update in Slide 24.
            2. Update the mixing_weights with the w_j update in Slide 24
            3. Update the covariance matrix with the sigma_j update in Slide 24.

        Slide 24 is in these slides: 
            https://github.com/NUCS349/nucs349.github.io/blob/master/lectures/eecs349_gaussian_mixture_models.pdf

        NOTE: When updating the parameters of the Gaussian you always use the output of
        the E step taken before this M step (e.g. update the means, mixing_weights, and covariances 
        simultaneously).

        Arguments:
            features {np.ndarray} -- Features to update means and covariances, given the
                current assignments.
            assignments {np.ndarray} -- Soft assignments of each point to one of the cluster,
                given by _e_step.

        Returns:
            means -- Updated means
            covariances -- Updated covariances
            mixing_weights -- Updated mixing weights
        """
        #Responsibility (Gamma) for one cluster = sum of posteriors of that cluster
        responsibility_vec = np.array([np.sum( assignments[i_cluster, :] ) for i_cluster in range(self.n_clusters)])

        #N = total number of points
        N = features.shape[0]

        #wj = responsibility/N
        wj = np.copy(responsibility_vec)
        wj = wj/N

        #miu_j = sum(gamma_j_i * x_i) / responsibility_j
        means = np.zeros((self.n_clusters, features.shape[1] ))
        sigma_copy = np.copy(self.covariances)

        for i_cluster in range(self.n_clusters):

            #cluster_vec = [[gamma_j_1 * x_1], [gamma_j_2 * x_2] ...]
            cluster_vec =  [ assignments[ i_cluster, i_feature] * features[i_feature, :] for i_feature in range(features.shape[0])]
            means[i_cluster,:] = np.sum( cluster_vec, axis=0 )/responsibility_vec[i_cluster]

            #sigma_j: sum of (gamma_ji * [x_i-miu_j][x_i-miu_j]^T) / responsibility (Gamma_j)
            #TODO: should we use the old means or the new means?
            miu_j = self.means[i_cluster, :].reshape(features.shape[1], 1)
            #TODO: not sure if this is the correct way?
            sigma_temp = np.zeros((features.shape[1], features.shape[1]))
            # sigma for one x_i
            for i_feature in range(features.shape[0]):
                x_i = features[i_feature, :].reshape(features.shape[1], 1)
                gamma_ji = assignments[i_cluster, i_feature]
                Gamma_j = responsibility_vec[i_cluster]
                sigma_temp_i = gamma_ji * (x_i - miu_j).dot((x_i - miu_j).T)/Gamma_j
                sigma_temp = sigma_temp + sigma_temp_i

            #update your sigma with the new sigma_temp value
            sigma_copy = self.update_covariance(sigma_copy, sigma_temp, i_cluster)

        # self.mixing_weights = wj
        # self.means = means
        # # self.covariances = sigma_copy
        # print("----------------")
        # print("assogments: ", assignments)
        # print("responsibility: ", )
        # print("N: ", N)
        # print("weights wj: ", wj)
        # print("means: ", means)
        # print("covariance", sigma_copy)
        return means, sigma_copy, wj

    def _init_covariance(self, n_features):
        """
        Initialize the covariance matrix given the covariance_type (spherical or
        diagonal). If spherical, each feature is treated the same (has equal covariance).
        If diagonal, each feature is treated independently (n_features covariances).

        Arguments:
            n_features {int} -- Number of features in the data for clustering

        Returns:
            [np.ndarray] -- Initial covariances (use np.random.rand)
        """
        if self.covariance_type == 'spherical':
            return np.random.rand(self.n_clusters)
        elif self.covariance_type == 'diagonal':
            return np.random.rand(self.n_clusters, n_features)

    def update_covariance(self, sigma_copy, sigma_temp, k_idx):
        """
        TODO:
        update a temporary copy of the covariance matrix, with the covariance matrix from the cluster with index k_idx
        Args:
            sigma_copy: copy of the covariance matrix
            sigma_temp: covariance matrix from the cluster with index k_idx
            k_idx: index of the cluster
        Return:
            updated copy of the covariance matrix
        """
        if self.covariance_type == "spherical":
            sigma_copy[k_idx] = np.mean(sigma_temp.diagonal())
        elif self.covariance_type == 'diagonal':
            sigma_copy[k_idx, :] = sigma_temp.diagonal()

        return sigma_copy

    def get_cluster_covariance(self, k_idx, n_features):
        """
        Compute the covariance matrix for a single cluster, based on the covariance type we'd like to enforce, and the index of the cluster.
        Args:
            k_idx: index of the cluster in self.covariance.
        Return:
            Covariance matrix of a cluster (n_features, n_features)
        """
        if self.covariance_type == "spherical":
            return np.identity(n_features) * self.covariances[k_idx]
        elif self.covariance_type == 'diagonal':
            cov = np.zeros((n_features, n_features))
            for i in range(n_features):
                cov[i, i] = self.covariances[k_idx, i]
            return cov

    def _log_likelihood(self, features, k_idx):
        """
        Compute the likelihood of the features given the index of the Gaussian
        in the mixture model. This function compute the log multivariate_normal
        distribution for features given the means and covariance of the ```k_idx```th
        Gaussian. To do this, you can use the function:

            scipy.stats.multivariate_normal.logpdf

        Read the documentation of this function to understand how it is used here:

            https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.multivariate_normal.html

        Once the raw likelihood is computed, incorporate the mixing_weights for the Gaussian
        via:

            log(mixing_weight) + logpdf

        Where logpdf is the output of multivariate_normal.

        Arguments:
            features {np.ndarray} -- Features to compute multivariate_normal distribution
                on.
            k_idx {int} -- Which Gaussian to use (e.g. use self.means[k_idx]

        Returns:
            np.ndarray -- log likelihoods of each feature given a Gaussian.  (all points)
        """

        # initialize return vector, which [n_samples, 1]
        ret_vec = np.zeros( features.shape[0] )

        # log(mixing_weight) + logpdf for each feature vector
        for i_vec in range(features.shape[0]):
            rv = multivariate_normal(self.means[k_idx, :], self.get_cluster_covariance(k_idx, features.shape[-1]))
            ret_vec[ i_vec ] = rv.logpdf( features[i_vec, :] ) + np.log( self.mixing_weights[k_idx] )    #use rv. logpdf

            if (self.mixing_weights[k_idx] == 0):
                pass
            #     print("---log weights: ", self.mixing_weights[k_idx])
            #     print("---log features: ", features[i_vec, :])

        return ret_vec


    def _overall_log_likelihood(self, features):
        denom = [
            self._log_likelihood(features, j) for j in range(self.n_clusters)
        ]
        return np.sum(denom)

    def _posterior(self, features, k):
        """
        Computes the posteriors given the log likelihoods for the GMM. Computes
        the posteriors for one of the Gaussians. To get all the posteriors, you have
        to iterate over this function. This function is implemented for you because the
        numerical issues can be tricky. We use the logsumexp trick to make it work (see
        below).

        Arguments:
            features {np.ndarray} -- Numpy array containing data (n_samples, n_features).
            k {int} -- Index of which Gaussian to compute posteriors for.

        Returns:
            np.ndarray -- Posterior probabilities for the selected Gaussian k, of size
                (n_samples,).
        """
        num = self._log_likelihood(features, k)
        denom = np.array([
            self._log_likelihood(features, j)
            for j in range(self.n_clusters)
        ])

        # Below is a useful function for safely computing large exponentials. It's a common
        # machine learning trick called the logsumexp trick:
        #   https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/

        max_value = denom.max(axis=0, keepdims=True)
        denom_sum = max_value + np.log(np.sum(np.exp(denom - max_value), axis=0))
        posteriors = np.exp(num - denom_sum)
        return posteriors

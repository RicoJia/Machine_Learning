import numpy as np

class KMeans():
    def __init__(self, n_clusters):
        """
        This class implements the traditional KMeans algorithm with hard assignments:

        https://en.wikipedia.org/wiki/K-means_clustering

        The KMeans algorithm has two steps:

        1. Update assignments
        2. Update the means

        While you only have to implement the fit and predict functions to pass the
        test cases, we recommend that you use an update_assignments function and an
        update_means function internally for the class.

        Use only numpy to implement this algorithm.

        Args:
            n_clusters (int): Number of clusters to cluster the given data into.

        """
        self.n_clusters = n_clusters
        self.means = None

    def fit(self, features):
        """
        Fit KMeans to the given data using `self.n_clusters` number of clusters.
        Features can have greater than 2 dimensions.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            None (saves model - means - internally)
        """
        self.means = self.init_means(features)      #(m, n_features)
        self.label_arr = np.array([])           #[n_samples]
        while True:                 # assume the model will converge
            self.cluster_dist = self.get_dist_to_mean(features)        #3D array (feature_num, mean_num)
            label_arr = self.label_features()
            if len(self.label_arr) != 0:  #if you're initializing it. update means
                if np.array_equal(self.label_arr, label_arr):
                    break
            self.label_arr = label_arr
            self.means = self.update_means(features)

    def update_means(self, features):
        """
        Update means after gettting new labels for each feature. 
        Args:
            features (np.ndarray): array containing inputs of size
            (n_samples, n_features).
        Return: 
             means [ num_means, n_features ]
        """
        #TODO
        means = np.copy(self.means)
        for i_mean in range(self.n_clusters):
            mean = means[i_mean, :]
            cluster = np.array( [ features[i, :] for i in range(features.shape[0]) if self.label_arr[i] == i_mean ] )

            for i_cln_mean in range(means.shape[1]):   #every column in mean
                means[i_mean, i_cln_mean] = np.mean(cluster[:, i_cln_mean] )

        return means

    def label_features(self):
        """
        Label each feauture based on their distance to each mean
        Return: table of labels (mean index) [num_feature]
        """
        label_arr = np.zeros((self.cluster_dist.shape[0]))
        for i_feature in range( self.cluster_dist.shape[0] ):
            label_arr[i_feature] = np.argmin(self.cluster_dist[i_feature, :])
        return  label_arr

    def get_dist_to_mean(self, features):
        """
        get the distance from each feature vector to a mean vector, given the index of the mean in self.means
        Args:
            features (np.ndarray): array containing inputs of size
            (n_samples, n_features).
        Return:
            table of distance from each feature point to a mean. [num_features, num_means]
        """
        cluster_dist = np.zeros((features.shape[0], self.means.shape[0]))
        for i_feature in range(features.shape[0]):
            for i_mean in range(self.means.shape[0]):
                cluster_dist[i_feature, i_mean] = np.linalg.norm((features[i_feature,:] - self.means[i_mean, :]))

        return cluster_dist

    def init_means(self, features):
        """
        Initialize k means randomly from the existing data
        Args:

        Return:
            means   (m, n_features)
        """

        index_arr = np.arange(features.shape[0])
        np.random.shuffle(index_arr)
        random_mean_index_arr = index_arr[:self.n_clusters]  #we assume n_clusters is way smaller than the feature space dimension
        means = np.array([ features[i, :] for i in random_mean_index_arr ])
        return means

    def predict(self, features):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict cluster
        membership labels.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted cluster membership for each features,
                of size (n_samples,). Each element of the array is the index of the
                cluster the sample belongs to.
        """
        label_arr = np.zeros( features.shape[0] )

        for i_feature in range( features.shape[0] ):
            feature = features[i_feature, :]
            means = np.zeros( self.n_clusters )
            for i_mean in range( self.n_clusters ):
                means[i_mean] = np.linalg.norm( self.means[i_mean,:] - feature)

            label_arr[i_feature] = np.argmin(means)

        return label_arr

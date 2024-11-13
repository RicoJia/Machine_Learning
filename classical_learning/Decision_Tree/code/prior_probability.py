import numpy as np


class PriorProbability:
    def __init__(self):
        """
        This is a simple classifier that only uses prior probability to classify
        points. It just looks at the classes for each data point and always predicts
        the most common class.

        """
        self.most_common_class = None

    def fit(self, features, targets):
        """
        Implement a classifier that works by prior probability. Takes in features
        and targets and fits the features to the targets using prior probability.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N
                examples.
        """
        self.most_common_class = self.find_mode(targets)

    def find_mode(self, targets):
        _targets = (np.copy(targets)).tolist()
        return max(set(_targets), key=_targets.count)

    def predict(self, data):
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            output: Nx1
        """
        return self.most_common_class * np.ones(data.shape[0])

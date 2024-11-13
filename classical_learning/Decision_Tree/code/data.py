import csv
import os

import numpy as np


def str_2_float(str):
    try:
        s = float(str)
        return s
    except ValueError:
        pass


def load_data(data_path):
    """
    Associated test case: tests/test_data.py

    Reading and manipulating data is a vital skill for machine learning.

    This function loads the data in data_path csv into two numpy arrays:
    features (of size NxK) and targets (of size Nx1) where N is the number of rows
    and K is the number of features.

    data_path leads to a csv comma-delimited file with each row corresponding to a
    different example. Each row contains binary features for each example
    (e.g. chocolate, fruity, caramel, etc.) The last column indicates the label for the
    example how likely it is to win a head-to-head matchup with another candy
    bar.

    This function reads in the csv file, and reads each row into two numpy arrays.
    The first array contains the features for each row. For example, in candy-data.csv
    the features are:

    chocolate,fruity,caramel,peanutyalmondy,nougat,crispedricewafer,hard,bar,pluribus

    The second array contains the targets for each row. The targets are in the last
    column of the csv file (labeled 'class'). The first row of the csv file contains
    the labels for each column and shouldn't be read into an array.

    Example:
    chocolate,fruity,caramel,peanutyalmondy,nougat,crispedricewafer,hard,bar,pluribus,class
    1,0,1,0,0,1,0,1,0,1

    should be turned into:

    [1,0,1,0,0,1,0,1,0] (features) and [1] (targets).

    This should be done for each row in the csv file, making arrays of size NxK and Nx1.

    Args:
        data_path (str): path to csv file containing the data

    Output:
        features (np.array): numpy array of size NxK containing the K features
        targets (np.array): numpy array of size Nx1 containing the 1 feature.
        attribute_names (list): list of strings containing names of each attribute
            (headers of csv)
    """
    attribute_names = []
    features = []
    targets = []
    line_index = 0
    with open(data_path) as data_file:
        reader = csv.reader(data_file, delimiter=",")
        for line in reader:
            if len(attribute_names) == 0:
                attribute_names = line
                attribute_names.pop()  # pop the last element which is 'class'
            else:
                line = [str_2_float(i) for i in line if str_2_float(i) is not None]
                targets.append(line[-1])
                line.pop()  # pop the last element of line
                features.append(line)
    # return np.array(features), np.array(targets).reshape(len(targets),1), attribute_names
    return np.array(features), np.array(targets), attribute_names


def train_test_split(features, targets, fraction):
    """
    Split features and targets into training and testing, randomly. N points from the data
    sampled for training and (features.shape[0] - N) points for testing. Where N:

        N = int(features.shape[0] * fraction)

    Returns train_features (size NxK), train_targets (Nx1), test_features (size MxK
    where M is the remaining points in data), and test_targets (Mx1).

    Args:
        features (np.array): numpy array containing features for each example
        targets (np.array): numpy array containing labels corresponding to each example.
        fraction (float between 0.0 and 1.0): fraction of examples to be drawn for training (SHOULD BE Randomly generated!!)

    Returns
        train_features: subset of features containing N examples to be used for training.
        train_targets: subset of targets corresponding to train_features containing targets.
        test_features: subset of features containing N examples to be used for testing.
        test_targets: subset of targets corresponding to test_features containing targets.
    """
    train_features = np.empty((0, features.shape[1]), float)
    train_targets = np.array([])
    test_features = np.empty((0, features.shape[1]), float)
    test_targets = np.array([])

    if fraction > 1.0:
        raise ValueError("N cannot be bigger than number of examples!")
    if fraction == 1.0:
        # when fraction = 1, returns the features and targets themselves
        return features, targets, features, targets

    N = int(features.shape[0] * fraction)

    train_row_number_set = set(
        np.random.choice(range(features.shape[0]), N, replace=False)
    )
    test_row_number_set = set(range(features.shape[0])) - train_row_number_set

    for train_index in train_row_number_set:
        train_features = np.vstack((train_features, features[train_index, :]))
        train_targets = np.append(train_targets, targets[train_index])

    for test_index in test_row_number_set:
        test_features = np.vstack((test_features, features[test_index, :]))
        test_targets = np.append(test_targets, targets[test_index])

    return train_features, train_targets, test_features, test_targets


# if __name__=='__main__':
#     # data_path = "../data/candy-data.csv"
#     # features, targets, attribute_names = load_data(data_path)
#     # train_features, train_targets, test_features, test_targets = train_test_split(features,targets,0.5)
#     # print "train: ", train_targets.shape[0], train_features.shape[0], " test: ", test_targets.shape[0], test_features.shape[0]
#     # print "split total: ", train_targets.shape[0] + test_features.shape[0]
#     # print "total: ", features.shape[0]
#     pass
